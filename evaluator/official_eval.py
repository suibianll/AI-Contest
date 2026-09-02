"""Canonical local HiF4 proxy evaluator.

``proxy-v2`` is deliberately a *local trend* evaluator, not a claim that the
hidden judge can be reproduced.  It preserves the published per-case score
formula and API contract while fixing the old proxy's two largest biases:

* the default panel is a deterministic stratified coverage of captured
  layer/role/window tensors (no Linear:Attention weighting or hash sampling);
* calibration state lifetime follows the judge call graph: one state per
  layer/role (Linear) and one per layer (Attention), while cases only vary the
  dynamic inputs.  This prevents a per-case calibration oracle.
* mechanism experiments reuse one immutable parent JSON and compare exact
  layer/role/window identities, so aggregate movement is separated into
  focus, unchanged-control, W/A or Q/K/V, worst-case, and timing deltas.

The hidden official tensors and Kunpeng hardware are not available locally.
All reports therefore label scores as ``proxy`` and local seconds as same-host
measurements.  Historical ``official-shape-v1`` artifacts remain immutable and
are not silently migrated.
"""

from __future__ import annotations

import argparse
import dataclasses
import gc
import hashlib
import importlib.util
import json
import math
import re
import statistics
import sys
import time
from pathlib import Path
from types import ModuleType
from typing import Any, Iterable, Mapping, Sequence

import torch

ROOT = Path(__file__).resolve().parents[1]
EVALUATOR_DIR = Path(__file__).resolve().parent
if str(EVALUATOR_DIR) not in sys.path:
    sys.path.insert(0, str(EVALUATOR_DIR))

from nvfp4_sim import nvfp4_encode  # noqa: E402
from reference_hif4 import (  # noqa: E402
    decode_standard_hif4,
    dequantize_hif4,
    dequantize_nvfp4,
    encode_standard_hif4,
    validate_hif4_params,
    validate_state,
)


PROTOCOL = "proxy-v2"
LEGACY_PROTOCOL = "official-shape-v1"
MODEL_NAME = "qwen2.5-0.5b"
MODEL_PATH = ROOT / "models" / MODEL_NAME
DATA_DIR = ROOT / "data" / "wikitext-2-raw-v1"
CACHE_DIR = ROOT / "artifacts" / "official_eval" / "cache"
DEFAULT_CACHE = CACHE_DIR / f"{MODEL_NAME}-{PROTOCOL}.pt"
CALIBRATION_LENGTHS = (10, 128, 512, 1024, 1024)
TEST_LENGTH = 128
TEST_LENGTHS = (10, 128, 512, 1024, 1024, 10, 128, 512, 1024, 1024, 128, 512)
TEST_WINDOW_COUNT = 12
# No artificial Linear:Attention weighting is applied.  The default case design
# is a deterministic stratified panel over real captured tensors: every
# layer/role appears once for Linear (the five holdout lengths rotate across
# the layer/role grid), and every layer appears once for each of the five
# official holdout lengths for Attention.  The full Cartesian expansion is
# still available with ``--full-cases`` as an explicit stress run.  The CLI
# limits below are opt-in smoke overrides only; they are recorded and must not
# be used for ranking.
LINEAR_CASE_COUNT: int | None = None
ATTENTION_CASE_COUNT: int | None = None
PANEL_WINDOW_INDICES = (0, 1, 2, 3, 4)
DEFAULT_CASE_DESIGN = "stratified-real-wa-panel-v1"
EFFECT_CASE_DESIGN = "paired-effect-panel-v1"
EFFECT_LINEAR_LAYERS = 8
FULL_CASE_DESIGN = "full-cartesian-real-wa-v3"
OFFICIAL_RUNTIME_LIMIT = 300.0
NVFP4_MODE = "amax6"
NVFP4_INPUT_CODEC = "e4m3-subnormal-ceil-v1"
WIKITEXT_REVISION = "b08601e04326c79dfdd32d625aee71d232d685c3"
WIKITEXT_CONFIG = "wikitext-2-raw-v1"
WIKITEXT_FILES = {
    "train": "train-00000-of-00001.parquet",
    "validation": "validation-00000-of-00001.parquet",
    "test": "test-00000-of-00001.parquet",
}
ROLES = ("q", "k", "v", "o", "fc_gate", "fc_up", "proj")
LINEAR_ROLE_FAMILIES = ("qkv", "o", "fc", "proj")
REQUIRED_APIS = (
    "hif4_calibration_and_quantize_weight",
    "hif4_dynamic_quantize_activation",
    "hif4_calibration_attention",
    "hif4_dynamic_quantize_q",
    "hif4_dynamic_quantize_k",
    "hif4_dynamic_quantize_v",
)

# These are user-confirmed official observations, used only for an explicit
# trend audit.  They never enter the proxy score or any candidate-side state.
# Comparing different historical scoring revisions would be misleading, so
# every entry carries a cohort and the audit only compares equal cohorts.
OFFICIAL_TREND_ANCHORS: dict[str, dict[str, Any]] = {
    "v084": {"score": 16517, "time_seconds": 252.563, "status": "pass", "cohort": "new-weight"},
    "v086": {"score": 16744, "time_seconds": 222.7, "status": "pass", "cohort": "new-weight"},
    "v138": {"score": 15715, "time_seconds": 208.0, "status": "pass", "cohort": "new-weight"},
    "v139": {"score": 15716, "time_seconds": 202.0, "status": "pass", "cohort": "new-weight"},
    "v140": {"score": 15838, "time_seconds": 207.0, "status": "pass", "cohort": "new-weight"},
    "v147": {"score": 16579, "time_seconds": 211.0, "status": "pass", "cohort": "new-weight"},
}


@dataclasses.dataclass(frozen=True)
class Window:
    split: str
    document_id: str
    row_start: int
    row_end: int
    token_start: int
    token_end: int
    input_ids: tuple[int, ...]


@dataclasses.dataclass(frozen=True)
class LinearCase:
    case_id: int
    layer: int
    role: str
    calibration_indices: tuple[int, ...]
    test_window: int


@dataclasses.dataclass(frozen=True)
class AttentionCase:
    case_id: int
    layer: int
    calibration_indices: tuple[int, ...]
    test_window: int


@dataclasses.dataclass
class RawPack:
    weights: list[dict[str, torch.Tensor]]
    calibration_activations: dict[str, list[list[torch.Tensor]]]
    test_activations: dict[str, list[list[torch.Tensor]]]
    calibration_qkv: list[list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]]
    test_qkv: list[list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]]
    calibration_windows: list[Window]
    test_windows: list[Window]
    layers: int
    hidden_size: int
    q_heads: int
    kv_heads: int
    head_dim: int
    metadata: dict[str, Any]
    # The public judge has no model-specific role names.  Qwen uses the
    # seven-role default below; cross-model probes may expose a different
    # operation set (for example GPT-2 has one GELU FFN input projection
    # instead of gated ``fc_gate``/``fc_up``).  Keep this optional for
    # backwards compatibility with existing Qwen cache payloads.
    roles: tuple[str, ...] = ROLES


@dataclasses.dataclass
class PreparedPack:
    weights: list[dict[str, tuple[torch.Tensor, torch.Tensor]]]
    linear_calibration_activations: dict[str, list[list[tuple[torch.Tensor, torch.Tensor]]]]
    test_activations: dict[str, list[list[tuple[torch.Tensor, torch.Tensor]]]]
    calibration_qkv: list[list[dict[str, tuple[torch.Tensor, torch.Tensor]]]]
    test_qkv: list[list[tuple[tuple[torch.Tensor, torch.Tensor], ...]]]
    calibration_windows: list[Window]
    linear_calibration_windows: list[Window]
    test_windows: list[Window]
    layers: int
    hidden_size: int
    q_heads: int
    kv_heads: int
    head_dim: int
    linear_cases: list[LinearCase]
    attention_cases: list[AttentionCase]
    metadata: dict[str, Any]
    roles: tuple[str, ...] = ROLES


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _cpu_float(value: torch.Tensor) -> torch.Tensor:
    return value.detach().to(device="cpu", dtype=torch.float32).contiguous()


def _flat(value: torch.Tensor) -> torch.Tensor:
    return _cpu_float(value).reshape(-1, value.shape[-1]).contiguous()


def _weight(module: torch.nn.Module) -> torch.Tensor:
    # HiF4 weight parameters use the module's native [out_features, in_features]
    # layout; the evaluator applies the transpose only in X @ W.T.
    return _cpu_float(module.weight).contiguous()


def _load_rows(path: Path) -> list[str]:
    try:
        import pyarrow.parquet as parquet
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("pyarrow is required for the pinned WikiText files") from exc
    table = parquet.read_table(path, columns=["text"])
    return [str(value) if value is not None else "" for value in table["text"].to_pylist()]


_TITLE_RE = re.compile(r"^\s*=+\s+.*?\s+=+\s*$")


def _documents(rows: Sequence[str], split: str) -> list[tuple[str, int, int, str]]:
    documents: list[tuple[str, int, int, str]] = []
    start: int | None = None
    title = "untitled"
    body: list[str] = []

    def flush(end_row: int) -> None:
        nonlocal start, title, body
        text = "\n".join(body).strip()
        if start is not None and text:
            documents.append((f"{split}:{start}:{title}", start, end_row, text))
        start = None
        title = "untitled"
        body = []

    for row, raw in enumerate(rows):
        text = raw.strip()
        if _TITLE_RE.match(text):
            flush(row - 1)
            start = row
            title = text
            body = [text]
        elif text:
            if start is None:
                start = row
            body.append(text)
    flush(len(rows) - 1)
    return documents


def _tokenized_documents(tokenizer: Any, rows: Sequence[str], split: str) -> list[tuple[str, int, int, list[int]]]:
    result: list[tuple[str, int, int, list[int]]] = []
    for document_id, row_start, row_end, text in _documents(rows, split):
        old_max = getattr(tokenizer, "model_max_length", None)
        if old_max is not None:
            tokenizer.model_max_length = 10**9
        try:
            encoded = tokenizer(text, add_special_tokens=False, return_attention_mask=False)
        finally:
            if old_max is not None:
                tokenizer.model_max_length = old_max
        ids = encoded["input_ids"]
        if ids and isinstance(ids[0], list):
            ids = ids[0]
        ids = [int(token) for token in ids]
        if ids:
            result.append((document_id, row_start, row_end, ids))
    return result


def _select_calibration_windows(tokenizer: Any, rows: Sequence[str]) -> list[Window]:
    docs = _tokenized_documents(tokenizer, rows, "train")
    selected: list[Window] = []
    used_ranges: dict[str, list[tuple[int, int]]] = {}
    for sample_index, length in enumerate(CALIBRATION_LENGTHS):
        # Pick a different source document for each calibration length when
        # possible.  The old evaluator always selected the first long article,
        # making the two Linear calibration samples nearly identical.
        candidates: list[Window] = []
        for document_id, row_start, row_end, ids in docs:
            if len(ids) < length:
                continue
            max_start = len(ids) - length
            starts = {
                0,
                max_start,
                int.from_bytes(
                    hashlib.sha256(f"cal:{sample_index}:{document_id}".encode("utf-8")).digest()[:8],
                    "big",
                ) % (max_start + 1),
            }
            for start in sorted(starts):
                end = start + length
                ranges = used_ranges.setdefault(document_id, [])
                if any(max(start, left) < min(end, right) for left, right in ranges):
                    continue
                candidates.append(Window("train", document_id, row_start, row_end, start, end, tuple(ids[start:end])))
        candidates.sort(key=lambda item: hashlib.sha256(repr(item).encode("utf-8")).hexdigest())
        found: Window | None = None
        # Prefer a document not used by an earlier calibration sample.  Fall
        # back to a fresh non-overlapping range if the split has too few long
        # documents.
        used_documents = {item.document_id for item in selected}
        for prefer_new_document in (True, False):
            for candidate in candidates:
                if prefer_new_document and candidate.document_id in used_documents:
                    continue
                found = candidate
                break
            if found is not None:
                break
        if found is None:
            raise RuntimeError(f"cannot select calibration window with length {length}")
        selected.append(found)
        used_ranges.setdefault(found.document_id, []).append((found.token_start, found.token_end))
    return selected


def _select_test_windows(tokenizer: Any, rows_by_split: Mapping[str, Sequence[str]]) -> list[Window]:
    """Select varied, document-balanced, variable-length holdout windows."""
    docs_by_split = {
        split: _tokenized_documents(tokenizer, rows_by_split[split], split)
        for split in ("validation", "test")
    }
    used_documents: dict[str, set[str]] = {"validation": set(), "test": set()}
    selected: list[Window] = []
    for index, length in enumerate(TEST_LENGTHS):
        split = "validation" if index % 2 == 0 else "test"
        candidates = [item for item in docs_by_split[split] if len(item[3]) >= length]
        candidates.sort(
            key=lambda item: hashlib.sha256(
                f"test:{index}:{split}:{item[0]}".encode("utf-8")
            ).hexdigest()
        )
        found: Window | None = None
        for prefer_new_document in (True, False):
            for document_id, row_start, row_end, ids in candidates:
                if prefer_new_document and document_id in used_documents[split]:
                    continue
                max_start = len(ids) - length
                start = int.from_bytes(
                    hashlib.sha256(f"offset:{index}:{document_id}".encode("utf-8")).digest()[:8],
                    "big",
                ) % (max_start + 1)
                found = Window(split, document_id, row_start, row_end, start, start + length, tuple(ids[start:start + length]))
                break
            if found is not None:
                break
        if found is None:
            raise RuntimeError(f"WikiText {split} split has no window of length {length}")
        selected.append(found)
        used_documents[split].add(found.document_id)
    return selected


def _apply_rope(q: torch.Tensor, k: torch.Tensor, rope: tuple[torch.Tensor, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb

    return apply_rotary_pos_emb(q, k, rope[0], rope[1])


def _load_qwen(device: torch.device) -> tuple[Any, torch.nn.Module]:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not MODEL_PATH.is_dir():
        raise FileNotFoundError(f"model directory does not exist: {MODEL_PATH}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True, use_fast=True)
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, local_files_only=True, torch_dtype=dtype)
    model.eval().to(device)
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False
    return tokenizer, model


def _capture_windows(model: torch.nn.Module, windows: Sequence[Window], device: torch.device) -> tuple[dict[str, list[list[torch.Tensor]]], list[list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]]]:
    blocks = list(model.model.layers)
    captures: list[dict[str, torch.Tensor]] = [{} for _ in blocks]
    rope: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    handles: list[Any] = []

    def capture_module(name: str, module: torch.nn.Module, layer_index: int, output: bool) -> None:
        def hook(_module: torch.nn.Module, inputs: tuple[Any, ...], value: Any, index: int = layer_index) -> None:
            if not inputs:
                raise RuntimeError(f"{name} received no inputs")
            captures[index][f"{name}_in"] = _cpu_float(inputs[0])
            if output:
                if isinstance(value, (tuple, list)):
                    value = value[0]
                captures[index][f"{name}_out"] = _cpu_float(value)

        handles.append(module.register_forward_hook(hook))

    for index, block in enumerate(blocks):
        attention = block.self_attn
        for name, module in (
            ("q", attention.q_proj),
            ("k", attention.k_proj),
            ("v", attention.v_proj),
            ("o", attention.o_proj),
            ("fc_gate", block.mlp.gate_proj),
            ("fc_up", block.mlp.up_proj),
            ("proj", block.mlp.down_proj),
        ):
            capture_module(name, module, index, name in {"q", "k", "v"})

    def rope_hook(_module: torch.nn.Module, _inputs: tuple[Any, ...], value: Any) -> None:
        if not isinstance(value, (tuple, list)) or len(value) != 2:
            raise RuntimeError("Qwen rotary hook returned an unexpected value")
        rope["value"] = (_cpu_float(value[0]), _cpu_float(value[1]))

    handles.append(model.model.rotary_emb.register_forward_hook(rope_hook))
    activations = {role: [] for role in ROLES}
    qkv_store: list[list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]] = []
    try:
        for window in windows:
            captures = [{} for _ in blocks]
            rope.clear()
            input_ids = torch.tensor(window.input_ids, dtype=torch.long, device=device).unsqueeze(0)
            with torch.no_grad():
                model(input_ids=input_ids, use_cache=False)
            if "value" not in rope:
                raise RuntimeError("Qwen forward did not expose rotary embeddings")
            per_layer_qkv: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
            per_layer_activations = {role: [] for role in ROLES}
            for index, captured in enumerate(captures):
                required = {f"{name}_in" for name in ROLES} | {"q_out", "k_out", "v_out"}
                missing = required - set(captured)
                if missing:
                    raise RuntimeError(f"layer {index} capture missing {sorted(missing)}")
                q = _flat(captured["q_out"]).reshape(1, -1, int(model.config.num_attention_heads), int(model.config.hidden_size) // int(model.config.num_attention_heads)).transpose(1, 2)
                k = _flat(captured["k_out"]).reshape(1, -1, int(model.config.num_key_value_heads), int(model.config.hidden_size) // int(model.config.num_attention_heads)).transpose(1, 2)
                v = _flat(captured["v_out"]).reshape(1, -1, int(model.config.num_key_value_heads), int(model.config.hidden_size) // int(model.config.num_attention_heads)).transpose(1, 2)
                q, k = _apply_rope(q, k, rope["value"])
                q_width = q.shape[1] * q.shape[-1]
                k_width = k.shape[1] * k.shape[-1]
                v_width = v.shape[1] * v.shape[-1]
                q = q.transpose(1, 2).reshape(-1, q_width).contiguous()
                k = k.transpose(1, 2).reshape(-1, k_width).contiguous()
                v = v.transpose(1, 2).reshape(-1, v_width).contiguous()
                per_layer_qkv.append((q, k, v))
                for role in ROLES:
                    per_layer_activations[role].append(_flat(captured[f"{role}_in"]))
            for role in ROLES:
                activations[role].append(per_layer_activations[role])
            qkv_store.append(per_layer_qkv)
    finally:
        for handle in handles:
            handle.remove()
    return activations, qkv_store


def capture_pack(device_name: str) -> RawPack:
    device = torch.device(device_name)
    tokenizer, model = _load_qwen(device)
    try:
        paths = {split: DATA_DIR / filename for split, filename in WIKITEXT_FILES.items()}
        missing = [str(path) for path in paths.values() if not path.is_file()]
        if missing:
            raise FileNotFoundError("missing pinned WikiText files: " + ", ".join(missing))
        train_rows = _load_rows(paths["train"])
        validation_rows = _load_rows(paths["validation"])
        test_rows = _load_rows(paths["test"])
        calibration_windows = _select_calibration_windows(tokenizer, train_rows)
        test_windows = _select_test_windows(
            tokenizer,
            {"validation": validation_rows, "test": test_rows},
        )
        blocks = list(model.model.layers)
        weights = []
        for block in blocks:
            attention = block.self_attn
            weights.append({
                "q": _weight(attention.q_proj),
                "k": _weight(attention.k_proj),
                "v": _weight(attention.v_proj),
                "o": _weight(attention.o_proj),
                "fc_gate": _weight(block.mlp.gate_proj),
                "fc_up": _weight(block.mlp.up_proj),
                "proj": _weight(block.mlp.down_proj),
            })
        cal_act, cal_qkv = _capture_windows(model, calibration_windows, device)
        test_act, test_qkv = _capture_windows(model, test_windows, device)
        config = model.config
        hidden = int(config.hidden_size)
        q_heads = int(config.num_attention_heads)
        kv_heads = int(config.num_key_value_heads)
        head_dim = int(getattr(config, "head_dim", hidden // q_heads))
        metadata = {
            "protocol": PROTOCOL,
            "model": MODEL_NAME,
            "linear_roles": list(ROLES),
            "model_path": str(MODEL_PATH),
            "model_revision": "Qwen/Qwen2.5-0.5B@060db6499f32faf8b98477b0a26969ef7d8b9987",
            "dataset": "Salesforce/wikitext",
            "dataset_config": WIKITEXT_CONFIG,
            "dataset_revision": WIKITEXT_REVISION,
            "calibration_lengths": list(CALIBRATION_LENGTHS),
            "test_length": TEST_LENGTH,
            "test_lengths": list(TEST_LENGTHS),
            "test_window_count": len(test_windows),
            "test_splits": sorted({window.split for window in test_windows}),
            "capture_device": str(device),
            "weights_dtype": "float32",
            "weight_layout": "[out_features, in_features]",
            "input_codec": NVFP4_INPUT_CODEC,
            "input_mode": NVFP4_MODE,
            "data_sha256": {split: sha256_file(path) for split, path in paths.items()},
        }
        return RawPack(
            weights, cal_act, test_act, cal_qkv, test_qkv,
            calibration_windows, test_windows, len(blocks), hidden,
            q_heads, kv_heads, head_dim, metadata,
        )
    finally:
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()


def save_pack(pack: RawPack, path: Path) -> None:
    payload = {
        "protocol": PROTOCOL,
        "weights": pack.weights,
        "calibration_activations": pack.calibration_activations,
        "test_activations": pack.test_activations,
        "calibration_qkv": pack.calibration_qkv,
        "test_qkv": pack.test_qkv,
        "calibration_windows": [dataclasses.asdict(window) for window in pack.calibration_windows],
        "test_windows": [dataclasses.asdict(window) for window in pack.test_windows],
        "layers": pack.layers,
        "hidden_size": pack.hidden_size,
        "q_heads": pack.q_heads,
        "kv_heads": pack.kv_heads,
        "head_dim": pack.head_dim,
        "roles": list(pack.roles),
        "metadata": pack.metadata,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def _window_from_payload(payload: Mapping[str, Any]) -> Window:
    return Window(
        str(payload["split"]), str(payload["document_id"]), int(payload["row_start"]),
        int(payload["row_end"]), int(payload["token_start"]), int(payload["token_end"]),
        tuple(int(token) for token in payload["input_ids"]),
    )


def load_pack(path: Path) -> RawPack:
    try:
        try:
            payload = torch.load(path, map_location="cpu", weights_only=True)
        except TypeError:
            payload = torch.load(path, map_location="cpu")
    except Exception as exc:
        raise RuntimeError(f"cannot read official data pack {path}: {exc}") from exc
    if not isinstance(payload, dict) or payload.get("protocol") != PROTOCOL:
        raise RuntimeError(
            f"data pack {path} is not protocol {PROTOCOL}; the old {LEGACY_PROTOCOL} "
            "cache is intentionally diagnostic-only"
        )
    calibration_windows = [_window_from_payload(item) for item in payload["calibration_windows"]]
    test_windows = [_window_from_payload(item) for item in payload["test_windows"]]
    if [len(window.input_ids) for window in calibration_windows] != list(CALIBRATION_LENGTHS):
        raise RuntimeError("calibration data pack has wrong variable-length schedule")
    if len(test_windows) != TEST_WINDOW_COUNT:
        raise RuntimeError(f"data pack has {len(test_windows)} test windows; expected {TEST_WINDOW_COUNT}")
    if [len(window.input_ids) for window in test_windows] != list(TEST_LENGTHS):
        raise RuntimeError("test data pack has wrong variable-length schedule")
    weights = payload["weights"]
    layers = int(payload["layers"])
    hidden_size = int(payload["hidden_size"])
    q_heads = int(payload["q_heads"])
    kv_heads = int(payload["kv_heads"])
    head_dim = int(payload["head_dim"])
    roles_payload = payload.get("roles", ROLES)
    if not isinstance(roles_payload, (list, tuple)):
        raise RuntimeError("data pack roles must be a list or tuple")
    roles = tuple(str(role) for role in roles_payload)
    if not roles or len(set(roles)) != len(roles) or not {"q", "k", "v"}.issubset(roles):
        raise RuntimeError("data pack roles must be unique and include q, k, v")
    # Reject stale snapshots produced by the pre-v1 transpose bug.  The public
    # API receives native [out_features, in_features] weights.
    if not isinstance(weights, list) or not weights:
        raise RuntimeError("official data pack has no layer weights")
    if len(weights) != layers:
        raise RuntimeError(f"data pack declares {layers} layers but stores {len(weights)} weight layers")
    expected_kv_width = kv_heads * head_dim
    for layer_index, per_layer in enumerate(weights):
        for role in roles:
            value = per_layer.get(role) if isinstance(per_layer, Mapping) else None
            if not torch.is_tensor(value) or value.ndim != 2:
                raise RuntimeError(f"layer {layer_index} weight {role} is not a matrix")
            rows, cols = map(int, value.shape)
            if role in {"k", "v"} and (rows, cols) != (expected_kv_width, hidden_size):
                raise RuntimeError(
                    f"layer {layer_index} weight {role} has {tuple(value.shape)}; "
                    f"expected native {(expected_kv_width, hidden_size)}"
                )
            if role in {"q", "o"} and (rows, cols) != (hidden_size, hidden_size):
                raise RuntimeError(
                    f"layer {layer_index} weight {role} has {tuple(value.shape)}; "
                    f"expected native {(hidden_size, hidden_size)}"
                )
        for role in set(roles) - {"q", "k", "v", "o"}:
            value = per_layer[role]
            rows, cols = map(int, value.shape)
            if role.startswith("fc_") and not (rows > cols and cols == hidden_size):
                raise RuntimeError(f"layer {layer_index} weight {role} is not native out-in layout")
            if role == "proj" and not (rows == hidden_size and cols > rows):
                raise RuntimeError(f"layer {layer_index} weight proj is not native out-in layout")
            if not role.startswith("fc_") and role != "proj":
                # Cross-model adapters may expose a differently named
                # expansive FFN operation (GPT-2 uses ``ffn_in``).  Keep only
                # the native out-in/input-width invariant here; the public
                # HiF4 validator performs the block-alignment check later.
                if cols != hidden_size:
                    raise RuntimeError(f"layer {layer_index} weight {role} has unsupported input width")
    metadata = dict(payload.get("metadata", {}))
    if metadata.get("weight_layout") not in {None, "[out_features, in_features]"}:
        raise RuntimeError("official data pack uses an unsupported weight layout")
    if metadata.get("input_codec") != NVFP4_INPUT_CODEC:
        raise RuntimeError(
            f"data pack uses input codec {metadata.get('input_codec')!r}; "
            f"expected {NVFP4_INPUT_CODEC!r}"
        )
    if metadata.get("input_mode") not in {None, NVFP4_MODE}:
        raise RuntimeError("data pack uses an unsupported NVFP4 input mode")

    def validate_activation_bank(
        bank_name: str,
        bank: Any,
        sample_count: int,
    ) -> None:
        if not isinstance(bank, Mapping):
            raise RuntimeError(f"{bank_name} is not a role mapping")
        for role in roles:
            samples = bank.get(role)
            if not isinstance(samples, list) or len(samples) != sample_count:
                raise RuntimeError(
                    f"{bank_name}[{role}] has {len(samples) if isinstance(samples, list) else 'invalid'} samples; "
                    f"expected {sample_count}"
                )
            for sample_index, per_layer in enumerate(samples):
                if not isinstance(per_layer, list) or len(per_layer) != layers:
                    raise RuntimeError(
                        f"{bank_name}[{role}][{sample_index}] has wrong layer count"
                    )
                for layer_index, tensor in enumerate(per_layer):
                    if not torch.is_tensor(tensor) or tensor.ndim != 2:
                        raise RuntimeError(
                            f"{bank_name}[{role}][{sample_index}][{layer_index}] is not a 2-D tensor"
                        )
                    expected_width = int(weights[layer_index][role].shape[1])
                    if int(tensor.shape[-1]) != expected_width:
                        raise RuntimeError(
                            f"{bank_name}[{role}][{sample_index}][{layer_index}] has input width "
                            f"{int(tensor.shape[-1])}; expected {expected_width}"
                        )

    def validate_qkv_bank(bank_name: str, bank: Any, sample_count: int) -> None:
        if not isinstance(bank, list) or len(bank) != sample_count:
            raise RuntimeError(f"{bank_name} has wrong sample count")
        expected_widths = (q_heads * head_dim, kv_heads * head_dim, kv_heads * head_dim)
        for sample_index, per_layer in enumerate(bank):
            if not isinstance(per_layer, list) or len(per_layer) != layers:
                raise RuntimeError(f"{bank_name}[{sample_index}] has wrong layer count")
            for layer_index, item in enumerate(per_layer):
                if not isinstance(item, (tuple, list)) or len(item) != 3:
                    raise RuntimeError(f"{bank_name}[{sample_index}][{layer_index}] is not a Q/K/V tuple")
                for name, tensor, expected_width in zip(("q", "k", "v"), item, expected_widths):
                    if not torch.is_tensor(tensor) or tensor.ndim != 2 or int(tensor.shape[-1]) != expected_width:
                        raise RuntimeError(
                            f"{bank_name}[{sample_index}][{layer_index}].{name} has wrong shape"
                        )

    validate_activation_bank(
        "calibration_activations", payload["calibration_activations"], len(calibration_windows)
    )
    validate_activation_bank(
        "test_activations", payload["test_activations"], len(test_windows)
    )
    validate_qkv_bank("calibration_qkv", payload["calibration_qkv"], len(calibration_windows))
    validate_qkv_bank("test_qkv", payload["test_qkv"], len(test_windows))
    window_keys = [
        (window.split, window.document_id, window.token_start, window.token_end)
        for window in test_windows
    ]
    if len(set(window_keys)) != len(window_keys):
        raise RuntimeError("test data pack contains duplicate holdout windows")
    if {window.split for window in test_windows} != {"validation", "test"}:
        raise RuntimeError("test data pack must contain both validation and test holdout windows")
    return RawPack(
        weights, payload["calibration_activations"], payload["test_activations"],
        payload["calibration_qkv"], payload["test_qkv"], calibration_windows, test_windows,
        layers, hidden_size, q_heads,
        kv_heads, head_dim, metadata, roles,
    )


def _pair(value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    carrier, scale = nvfp4_encode(value, NVFP4_MODE)
    return carrier.contiguous(), scale.contiguous()


def _depth_spread_indices(total: int, count: int) -> tuple[int, ...]:
    """Choose deterministic indices spanning the full model depth."""
    if total <= 0 or count <= 0:
        return ()
    if count >= total:
        return tuple(range(total))
    if count == 1:
        return (0,)
    values = {
        int(round(index * (total - 1) / (count - 1)))
        for index in range(count)
    }
    return tuple(sorted(values))


def _choose_cases(
    pack: RawPack,
    linear_count: int | None = LINEAR_CASE_COUNT,
    attention_count: int | None = ATTENTION_CASE_COUNT,
    full_cases: bool = False,
    effect_panel: bool = False,
) -> tuple[list[LinearCase], list[AttentionCase]]:
    """Select a deterministic real-W/A panel; optional limits are smoke-only.

    The default covers every Linear layer/role exactly once, with the five
    official holdout lengths rotated across the grid, and every Attention
    layer once for each of those five lengths.  ``full_cases`` explicitly
    expands to every layer/role/window tuple for stress testing.
    ``effect_panel`` selects eight depth-spread layers with every Linear role
    plus five depth/length-spread Attention sentinels for paired iteration.
    All modes use captured tensors; ``linear_count`` and ``attention_count``
    are only development prefix limits and are never a ranking panel.
    """
    if len(pack.test_windows) == 0 or len(pack.calibration_windows) == 0:
        raise RuntimeError("case pool has no calibration/test windows")
    linear_calibration_indices = tuple(range(min(2, len(pack.calibration_windows))))
    if not linear_calibration_indices:
        raise RuntimeError("case pool has no Linear calibration windows")
    roles = tuple(getattr(pack, "roles", ROLES))
    if full_cases and effect_panel:
        raise ValueError("full_cases and effect_panel are mutually exclusive")
    if effect_panel and (linear_count is not None or attention_count is not None):
        raise ValueError("effect_panel cannot be combined with case prefix limits")
    if full_cases:
        linear_pool = [
            (layer, role, test_window)
            for test_window in range(len(pack.test_windows))
            for layer in range(pack.layers)
            for role in roles
        ]
        attention_pool = [
            (layer, test_window)
            for test_window in range(len(pack.test_windows))
            for layer in range(pack.layers)
        ]
    else:
        panel_windows = tuple(
            index for index in PANEL_WINDOW_INDICES if index < len(pack.test_windows)
        )
        if not panel_windows:
            raise RuntimeError("case pool has no windows in the default panel")
        if effect_panel:
            # Iteration panel: every selected depth covers all static Linear
            # roles, while five Attention sentinels span both depth and the
            # published sequence lengths.  Calibration still follows the
            # full judge call graph; only dynamic scoring cases are reduced.
            linear_layers = _depth_spread_indices(
                pack.layers, min(EFFECT_LINEAR_LAYERS, pack.layers)
            )
            attention_layers = _depth_spread_indices(
                pack.layers, min(len(panel_windows), pack.layers)
            )
            linear_pool = [
                (
                    layer,
                    role,
                    panel_windows[(layer + role_index) % len(panel_windows)],
                )
                for layer in linear_layers
                for role_index, role in enumerate(roles)
            ]
            attention_pool = [
                (
                    attention_layers[
                        min(
                            len(attention_layers) - 1,
                            int(round(index * (len(attention_layers) - 1) / max(1, len(panel_windows) - 1))),
                        )
                    ],
                    test_window,
                )
                for index, test_window in enumerate(panel_windows)
            ]
        else:
            # Each layer sees the same length coverage, while the
            # role-to-window assignment rotates with layer index.  This
            # avoids coupling one role permanently to one document/split and
            # avoids hash-based sampling.
            linear_pool = [
                (
                    layer,
                    role,
                    panel_windows[(layer + role_index) % len(panel_windows)],
                )
                for layer in range(pack.layers)
                for role_index, role in enumerate(roles)
            ]
            attention_pool = [
                (layer, test_window)
                for test_window in panel_windows
                for layer in range(pack.layers)
            ]
    if linear_count is not None:
        if linear_count <= 0:
            raise ValueError("linear_count must be positive when supplied")
        linear_pool = linear_pool[: min(int(linear_count), len(linear_pool))]
    if attention_count is not None:
        if attention_count <= 0:
            raise ValueError("attention_count must be positive when supplied")
        attention_pool = attention_pool[: min(int(attention_count), len(attention_pool))]
    linear_cases = [
        LinearCase(case_id=index, layer=layer, role=role,
                   calibration_indices=linear_calibration_indices,
                   test_window=test_window)
        for index, (layer, role, test_window) in enumerate(linear_pool)
    ]
    attention_cases = [
        AttentionCase(case_id=index, layer=layer,
                      calibration_indices=tuple(range(len(pack.calibration_windows))),
                      test_window=test_window)
        for index, (layer, test_window) in enumerate(attention_pool)
    ]
    return linear_cases, attention_cases


def prepare_pack(
    raw: RawPack,
    linear_count: int | None = LINEAR_CASE_COUNT,
    attention_count: int | None = ATTENTION_CASE_COUNT,
    full_cases: bool = False,
    effect_panel: bool = False,
) -> PreparedPack:
    weights = [{role: _pair(value) for role, value in per_layer.items()} for per_layer in raw.weights]
    # Keep every calibration window available for Attention.  Linear follows
    # the public call graph: one calibration state per layer/role, using the
    # first two explicitly designated Linear folds.  Test cases may be fewer
    # than the official panel, but they must not create a fresh state per test
    # tuple (that would give output-aware candidates an unfair per-case oracle).
    linear_calibration_indices = tuple(range(min(2, len(raw.calibration_windows))))
    if not linear_calibration_indices:
        raise RuntimeError("data pack has no Linear calibration windows")
    linear_calibration_windows = [raw.calibration_windows[index] for index in linear_calibration_indices]
    roles = tuple(raw.roles)
    cal_act = {role: [[_pair(raw.calibration_activations[role][sample][layer]) for layer in range(raw.layers)] for sample in range(len(raw.calibration_windows))] for role in roles}
    test_act = {role: [[_pair(raw.test_activations[role][sample][layer]) for layer in range(raw.layers)] for sample in range(len(raw.test_windows))] for role in roles}
    cal_qkv = [[{"q": _pair(q), "k": _pair(k), "v": _pair(v)} for q, k, v in per_layer] for per_layer in raw.calibration_qkv]
    test_qkv = [[(_pair(q), _pair(k), _pair(v)) for q, k, v in per_layer] for per_layer in raw.test_qkv]
    linear_cases, attention_cases = _choose_cases(
        raw,
        linear_count,
        attention_count,
        full_cases=full_cases,
        effect_panel=effect_panel,
    )
    metadata = dict(raw.metadata)
    metadata.update({
        "linear_case_count": len(linear_cases),
        "attention_case_count": len(attention_cases),
        "nvfp4_mode": NVFP4_MODE,
        "input_codec": NVFP4_INPUT_CODEC,
        "case_design": (
            FULL_CASE_DESIGN
            if full_cases and linear_count is None and attention_count is None
            else EFFECT_CASE_DESIGN
            if effect_panel and linear_count is None and attention_count is None
            else DEFAULT_CASE_DESIGN
            if not full_cases and linear_count is None and attention_count is None
            else "explicit-smoke-prefix-v4"
        ),
        "panel_window_indices": list(PANEL_WINDOW_INDICES),
        "full_cases": full_cases,
        "effect_panel": effect_panel,
        "linear_case_limit": linear_count,
        "attention_case_limit": attention_count,
        "linear_calibration_indices": list(linear_calibration_indices),
        "calibration_call_graph": "all-layer-role-once; attention-layer-once",
        "linear_roles": list(roles),
    })
    return PreparedPack(
        weights, cal_act, test_act, cal_qkv, test_qkv,
        raw.calibration_windows, linear_calibration_windows, raw.test_windows, raw.layers, raw.hidden_size,
        raw.q_heads, raw.kv_heads, raw.head_dim, linear_cases, attention_cases, metadata, roles,
    )


def _evaluation_scope(pack: PreparedPack) -> dict[str, Any]:
    """Classify a run so reports cannot mix ranking, paired, and smoke numbers."""
    design = str(pack.metadata.get("case_design", "unknown"))
    linear_cases = len(pack.linear_cases)
    attention_cases = len(pack.attention_cases)
    default_counts = pack.layers * len(pack.roles), pack.layers * len(PANEL_WINDOW_INDICES)
    if design == DEFAULT_CASE_DESIGN and (linear_cases, attention_cases) == default_counts:
        return {
            "kind": "default-panel",
            "intent": "proxy-ranking-within-identical-cache",
            "comparable_for_proxy_ranking": True,
            "paired_only": False,
            "stress_only": False,
            "smoke_only": False,
            "official_score_equivalent": False,
        }
    if design == EFFECT_CASE_DESIGN:
        return {
            "kind": "effect-panel",
            "intent": "paired-mechanism-diagnosis",
            "comparable_for_proxy_ranking": False,
            "paired_only": True,
            "stress_only": False,
            "smoke_only": False,
            "official_score_equivalent": False,
        }
    if design == FULL_CASE_DESIGN:
        return {
            "kind": "full-stress",
            "intent": "stress-only-regression-check",
            "comparable_for_proxy_ranking": False,
            "paired_only": False,
            "stress_only": True,
            "smoke_only": False,
            "official_score_equivalent": False,
        }
    if design == "explicit-smoke-prefix-v4":
        return {
            "kind": "smoke-prefix",
            "intent": "interface-and-local-sanity-only",
            "comparable_for_proxy_ranking": False,
            "paired_only": False,
            "stress_only": False,
            "smoke_only": True,
            "official_score_equivalent": False,
        }
    return {
        "kind": "unknown",
        "intent": "do-not-compare",
        "comparable_for_proxy_ranking": False,
        "paired_only": False,
        "stress_only": False,
        "smoke_only": False,
        "official_score_equivalent": False,
    }


def load_solution(path: Path) -> ModuleType:
    source = path.resolve()
    module_name = f"_hif4_official_{hashlib.sha1(str(source).encode()).hexdigest()}"
    spec = importlib.util.spec_from_file_location(module_name, source)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load solution: {source}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    missing = [name for name in REQUIRED_APIS if not callable(getattr(module, name, None))]
    if missing:
        raise AttributeError(f"solution is missing functions: {', '.join(missing)}")
    return module


def _move_pair(pair: tuple[torch.Tensor, torch.Tensor], device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    return pair[0].to(device), pair[1].to(device)


def _move_qkv(item: Mapping[str, tuple[torch.Tensor, torch.Tensor]], device: torch.device) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    return {name: _move_pair(pair, device) for name, pair in item.items()}


def _attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, q_heads: int, kv_heads: int, head_dim: int) -> torch.Tensor:
    batch, tokens, _ = q.shape
    qh = q.reshape(batch, tokens, q_heads, head_dim).transpose(1, 2)
    group = q_heads // kv_heads
    kh = k.reshape(batch, tokens, kv_heads, head_dim).transpose(1, 2).repeat_interleave(group, dim=1)
    vh = v.reshape(batch, tokens, kv_heads, head_dim).transpose(1, 2).repeat_interleave(group, dim=1)
    probabilities = torch.softmax(qh @ kh.transpose(-1, -2) / math.sqrt(head_dim), dim=-1)
    return (probabilities @ vh).transpose(1, 2).reshape(batch, tokens, q_heads * head_dim)


def _attention_trace(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q_heads: int,
    kv_heads: int,
    head_dim: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return attention output plus logits and probabilities for diagnostics.

    The public score only needs the output.  Keeping the intermediate tensors
    behind this small helper lets the evaluator identify whether a candidate
    loses information in Q/K logits, softmax probabilities, or V/output
    reconstruction without changing any candidate API call.
    """
    batch, tokens, _ = q.shape
    qh = q.reshape(batch, tokens, q_heads, head_dim).transpose(1, 2)
    group = q_heads // kv_heads
    kh = k.reshape(batch, tokens, kv_heads, head_dim).transpose(1, 2).repeat_interleave(group, dim=1)
    vh = v.reshape(batch, tokens, kv_heads, head_dim).transpose(1, 2).repeat_interleave(group, dim=1)
    logits = qh @ kh.transpose(-1, -2) / math.sqrt(head_dim)
    probabilities = torch.softmax(logits, dim=-1)
    output = (probabilities @ vh).transpose(1, 2).reshape(batch, tokens, q_heads * head_dim)
    return output, logits, probabilities


def _score(standard: torch.Tensor, player: torch.Tensor, reference: torch.Tensor) -> float:
    return _score_details(standard, player, reference)["gain"]


def _score_details(
    standard: torch.Tensor,
    player: torch.Tensor,
    reference: torch.Tensor,
) -> dict[str, float]:
    standard_mse = float((standard - reference).square().mean())
    player_mse = float((player - reference).square().mean())
    reference_energy = float(reference.square().mean())
    if not math.isfinite(standard_mse) or standard_mse <= 0.0:
        raise ValueError("MSE_STD must be finite and positive")
    if not math.isfinite(player_mse):
        raise ValueError("MSE_PLAYER must be finite")
    if not math.isfinite(reference_energy) or reference_energy <= 0.0:
        raise ValueError("reference energy must be finite and positive")
    return {
        "mse_standard": standard_mse,
        "mse_player": player_mse,
        "reference_energy": reference_energy,
        "relative_player_mse": player_mse / reference_energy,
        "gain": (standard_mse - player_mse) / standard_mse,
    }


def _mse(value: torch.Tensor, reference: torch.Tensor) -> float:
    result = float((value - reference).square().mean())
    if not math.isfinite(result):
        raise ValueError("diagnostic MSE must be finite")
    return result


def _relative_mse(value: torch.Tensor, reference: torch.Tensor) -> float:
    energy = float(reference.square().mean())
    if not math.isfinite(energy) or energy <= 0.0:
        raise ValueError("diagnostic reference energy must be finite and positive")
    return _mse(value, reference) / energy


def _linear_error_source_details(
    standard: torch.Tensor,
    weight_only: torch.Tensor,
    activation_only: torch.Tensor,
    both: torch.Tensor,
    reference: torch.Tensor,
    ref_weight: torch.Tensor,
    standard_weight: torch.Tensor,
    player_weight: torch.Tensor,
    ref_activation: torch.Tensor,
    standard_activation: torch.Tensor,
    player_activation: torch.Tensor,
) -> dict[str, float]:
    """Four-arm Linear output decomposition.

    ``standard`` is the fixed legal reference codec, ``both`` is the submitted
    pair, and the two middle arms replace exactly one operand.  The interaction
    term is reported in the conventional error sign and as a gain where a
    positive value means the two candidate encoders provide super-additive
    complementarity; a negative value means overlapping or diminishing returns.
    """
    details = _score_details(standard, both, reference)
    e00 = details["mse_standard"]
    e10 = _mse(weight_only, reference)
    e01 = _mse(activation_only, reference)
    e11 = details["mse_player"]
    details.update({
        "mse_w_only": e10,
        "mse_a_only": e01,
        "mse_both": e11,
        "gain_w_only": (e00 - e10) / e00,
        "gain_a_only": (e00 - e01) / e00,
        "gain_both": (e00 - e11) / e00,
        "interaction_mse": e11 - e10 - e01 + e00,
        "interaction_gain": (e10 + e01 - e00 - e11) / e00,
        "weight_relative_mse": _relative_mse(player_weight, ref_weight),
        "activation_relative_mse": _relative_mse(player_activation, ref_activation),
        "standard_weight_relative_mse": _relative_mse(standard_weight, ref_weight),
        "standard_activation_relative_mse": _relative_mse(standard_activation, ref_activation),
    })
    return details


def _attention_error_source_details(
    standard: torch.Tensor,
    q_only: torch.Tensor,
    k_only: torch.Tensor,
    v_only: torch.Tensor,
    qk_only: torch.Tensor,
    both: torch.Tensor,
    reference: torch.Tensor,
    reference_logits: torch.Tensor,
    standard_logits: torch.Tensor,
    player_logits: torch.Tensor,
    reference_probabilities: torch.Tensor,
    standard_probabilities: torch.Tensor,
    player_probabilities: torch.Tensor,
) -> dict[str, float]:
    """Attention Q/K/V output and intermediate decomposition."""
    details = _score_details(standard, both, reference)
    e000 = details["mse_standard"]
    e100 = _mse(q_only, reference)
    e010 = _mse(k_only, reference)
    e001 = _mse(v_only, reference)
    e110 = _mse(qk_only, reference)
    e111 = details["mse_player"]
    eps = torch.finfo(reference_probabilities.dtype).tiny
    ref_prob = reference_probabilities.clamp_min(eps)
    player_prob = player_probabilities.clamp_min(eps)
    standard_prob = standard_probabilities.clamp_min(eps)
    details.update({
        "mse_q_only": e100,
        "mse_k_only": e010,
        "mse_v_only": e001,
        "mse_qk_only": e110,
        "mse_both": e111,
        "gain_q_only": (e000 - e100) / e000,
        "gain_k_only": (e000 - e010) / e000,
        "gain_v_only": (e000 - e001) / e000,
        "gain_qk_only": (e000 - e110) / e000,
        "gain_both": (e000 - e111) / e000,
        "qk_interaction_mse": e110 - e100 - e010 + e000,
        "qk_interaction_gain": (e100 + e010 - e000 - e110) / e000,
        "qkv_interaction_mse": e111 - e110 - e001 + e000,
        "qkv_interaction_gain": (e110 + e001 - e000 - e111) / e000,
        "logit_mse_standard": _mse(standard_logits, reference_logits),
        "logit_mse_player": _mse(player_logits, reference_logits),
        "probability_mse_standard": _mse(standard_probabilities, reference_probabilities),
        "probability_mse_player": _mse(player_probabilities, reference_probabilities),
        "probability_kl_standard_to_reference": float(
            (ref_prob * (ref_prob.log() - standard_prob.log())).mean()
        ),
        "probability_kl_player_to_reference": float(
            (ref_prob * (ref_prob.log() - player_prob.log())).mean()
        ),
    })
    return details


def _mean_metric(items: Sequence[Mapping[str, Any]], key: str) -> float:
    values = [float(item[key]) for item in items if key in item]
    return sum(values) / max(1, len(values))


def _linear_decomposition_summary(items: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    summary = {
        "case_count": len(items),
        "mse": {
            "standard": _mean_metric(items, "mse_standard"),
            "w_only": _mean_metric(items, "mse_w_only"),
            "a_only": _mean_metric(items, "mse_a_only"),
            "both": _mean_metric(items, "mse_both"),
        },
        "gain": {
            "w_only": _mean_metric(items, "gain_w_only"),
            "a_only": _mean_metric(items, "gain_a_only"),
            "both": _mean_metric(items, "gain_both"),
            "interaction": _mean_metric(items, "interaction_gain"),
        },
        "operand_relative_mse": {
            "weight": _mean_metric(items, "weight_relative_mse"),
            "activation": _mean_metric(items, "activation_relative_mse"),
            "standard_weight": _mean_metric(items, "standard_weight_relative_mse"),
            "standard_activation": _mean_metric(items, "standard_activation_relative_mse"),
        },
    }
    gains = summary["gain"]
    if gains["w_only"] < 0.0 and gains["a_only"] < 0.0 and gains["both"] > 0.0:
        summary["interpretation"] = "paired_coordinate_coupling_likely"
    elif gains["w_only"] > gains["a_only"]:
        summary["interpretation"] = "weight_dominant"
    elif gains["a_only"] > gains["w_only"]:
        summary["interpretation"] = "activation_dominant"
    else:
        summary["interpretation"] = "mixed_or_neutral"
    return summary


def _attention_decomposition_summary(items: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    summary = {
        "case_count": len(items),
        "mse": {
            "standard": _mean_metric(items, "mse_standard"),
            "q_only": _mean_metric(items, "mse_q_only"),
            "k_only": _mean_metric(items, "mse_k_only"),
            "v_only": _mean_metric(items, "mse_v_only"),
            "qk_only": _mean_metric(items, "mse_qk_only"),
            "both": _mean_metric(items, "mse_both"),
        },
        "gain": {
            "q_only": _mean_metric(items, "gain_q_only"),
            "k_only": _mean_metric(items, "gain_k_only"),
            "v_only": _mean_metric(items, "gain_v_only"),
            "qk_only": _mean_metric(items, "gain_qk_only"),
            "both": _mean_metric(items, "gain_both"),
            "qk_interaction": _mean_metric(items, "qk_interaction_gain"),
            "qkv_interaction": _mean_metric(items, "qkv_interaction_gain"),
        },
        "intermediate": {
            "logit_mse_standard": _mean_metric(items, "logit_mse_standard"),
            "logit_mse_player": _mean_metric(items, "logit_mse_player"),
            "probability_mse_standard": _mean_metric(items, "probability_mse_standard"),
            "probability_mse_player": _mean_metric(items, "probability_mse_player"),
            "probability_kl_standard_to_reference": _mean_metric(
                items, "probability_kl_standard_to_reference"
            ),
            "probability_kl_player_to_reference": _mean_metric(
                items, "probability_kl_player_to_reference"
            ),
        },
    }
    gains = summary["gain"]
    if gains["q_only"] < 0.0 and gains["k_only"] < 0.0 and gains["qk_only"] > 0.0:
        summary["interpretation"] = "paired_qk_coupling_likely"
    elif gains["qk_only"] > gains["v_only"]:
        summary["interpretation"] = "qk_dominant"
    elif gains["v_only"] > max(gains["q_only"], gains["k_only"]):
        summary["interpretation"] = "v_dominant"
    else:
        summary["interpretation"] = "mixed_or_neutral"
    return summary


def _group_summary(
    items: Sequence[Mapping[str, Any]],
    key: str,
    summary_fn: Any,
) -> dict[str, Any]:
    groups: dict[str, list[Mapping[str, Any]]] = {}
    for item in items:
        groups.setdefault(str(item[key]), []).append(item)
    return {name: summary_fn(group) for name, group in sorted(groups.items())}


def _linear_role_family(role: str) -> str:
    """Map model-specific static Linear roles to comparable families.

    The public judge does not expose role names, and cross-model probes can
    use ``ffn_in`` instead of gated ``fc_gate``/``fc_up``.  Keeping this map in
    the evaluator makes the diagnostic grouping explicit without changing the
    candidate API or pretending that static Q/K/V are dynamic Attention Q/K/V.
    """
    if role in {"q", "k", "v"}:
        return "qkv"
    if role == "o":
        return "o"
    if role == "proj":
        return "proj"
    if role.startswith("fc_") or role == "ffn_in":
        return "fc"
    return role


def _delta_summary(values: Sequence[float]) -> dict[str, Any]:
    """Summarize a candidate-vs-baseline signed role delta."""
    if not values:
        return {
            "case_count": 0,
            "mean_delta_gain": 0.0,
            "positive_cases": 0,
            "negative_cases": 0,
            "zero_cases": 0,
            "min_delta_gain": 0.0,
            "max_delta_gain": 0.0,
        }
    return {
        "case_count": len(values),
        "mean_delta_gain": sum(values) / len(values),
        "positive_cases": sum(value > 0.0 for value in values),
        "negative_cases": sum(value < 0.0 for value in values),
        "zero_cases": sum(value == 0.0 for value in values),
        "min_delta_gain": min(values),
        "max_delta_gain": max(values),
    }


def _linear_case_identity(item: Mapping[str, Any]) -> tuple[Any, ...]:
    """Return the stable identity used to pair cases across candidates."""
    return (
        int(item.get("layer", -1)),
        str(item.get("role", "")),
        int(item.get("test_window", -1)),
        str(item.get("test_split", "")),
        int(item.get("test_length", -1)),
    )


def _linear_candidate_role_diagnostics(
    results: Sequence[Mapping[str, Any]],
    baseline_name: str | None = None,
) -> dict[str, Any]:
    """Compare static Linear roles across candidates on one shared panel.

    The per-candidate decomposition already reports W/A arms against the
    independent standard codec.  This second diagnostic answers a different
    question: ``candidate gain - baseline gain`` for the exact same
    layer/role/window.  It is intentionally evaluator-only and never changes
    a candidate score or invokes an additional public API.
    """
    eligible = [
        result for result in results
        if result.get("status") == "ok"
        and result.get("decomposition", {}).get("linear", {}).get("enabled")
    ]
    if len(eligible) < 2:
        return {
            "enabled": False,
            "reason": "at least two successful candidates with Linear decomposition are required",
        }
    by_name = {str(result.get("candidate", "")): result for result in eligible}
    if baseline_name is not None:
        baseline = by_name.get(str(baseline_name))
        if baseline is None:
            return {
                "enabled": False,
                "reason": f"requested baseline {baseline_name!r} is not an eligible candidate",
            }
    else:
        # v086 (often called v86 in notes) is the verified Linear/Attention
        # parent in the current cohort.  Keep both spellings because archive
        # manifests use the zero-padded form while ad-hoc runs often do not.
        baseline = next(
            (by_name[name] for name in ("v086", "v86") if name in by_name),
            eligible[0],
        )
    baseline_label = str(baseline.get("candidate", "baseline"))
    baseline_cases = {
        _linear_case_identity(item): item
        for item in baseline.get("case_scores", {}).get("linear", [])
    }
    if not baseline_cases:
        return {"enabled": False, "reason": "baseline has no Linear case scores"}

    candidate_payload: dict[str, Any] = {}
    for result in eligible:
        name = str(result.get("candidate", "candidate"))
        if name == baseline_label:
            continue
        paired: list[dict[str, Any]] = []
        for item in result.get("case_scores", {}).get("linear", []):
            baseline_item = baseline_cases.get(_linear_case_identity(item))
            if baseline_item is None:
                continue
            role = str(item.get("role", ""))
            paired.append({
                "layer": int(item.get("layer", -1)),
                "role": role,
                "role_family": str(item.get("role_family", _linear_role_family(role))),
                "test_window": int(item.get("test_window", -1)),
                "delta_gain": float(item.get("gain", 0.0)) - float(baseline_item.get("gain", 0.0)),
            })
        if not paired:
            candidate_payload[name] = {
                "case_count": 0,
                "by_role": {},
                "by_role_family": {},
                "by_role_layer": {},
                "worst_cases": [],
            }
            continue

        def grouped(key: str) -> dict[str, dict[str, Any]]:
            groups: dict[str, list[float]] = {}
            for item in paired:
                groups.setdefault(str(item[key]), []).append(float(item["delta_gain"]))
            return {group: _delta_summary(values) for group, values in sorted(groups.items())}

        role_layer: dict[str, dict[str, Any]] = {}
        for item in paired:
            key = f"{item['role']}@layer{item['layer']}"
            role_layer.setdefault(key, {"role": item["role"], "role_family": item["role_family"], "layer": item["layer"], "values": []})
            role_layer[key]["values"].append(float(item["delta_gain"]))
        role_layer = {
            key: {
                **{field: value for field, value in value.items() if field != "values"},
                **_delta_summary(value["values"]),
            }
            for key, value in role_layer.items()
        }
        candidate_payload[name] = {
            "case_count": len(paired),
            "by_role": grouped("role"),
            "by_role_family": grouped("role_family"),
            "by_role_layer": role_layer,
            "worst_cases": [
                {
                    "layer": item["layer"],
                    "role": item["role"],
                    "role_family": item["role_family"],
                    "test_window": item["test_window"],
                    "delta_gain": item["delta_gain"],
                }
                for item in sorted(paired, key=lambda value: value["delta_gain"])[:16]
            ],
        }
    return {
        "enabled": True,
        "baseline": baseline_label,
        "formula": "candidate case gain - baseline case gain; positive means candidate improves the same layer/role/window",
        "candidates": candidate_payload,
    }


def _attention_case_identity(item: Mapping[str, Any]) -> tuple[Any, ...]:
    """Return the stable identity used to pair Attention cases."""
    return (
        int(item.get("layer", -1)),
        int(item.get("test_window", -1)),
        str(item.get("test_split", "")),
        int(item.get("test_length", -1)),
    )


def _effect_label(positive: int, negative: int, zero: int) -> str:
    """Describe paired signs without introducing a promotion threshold."""
    if positive == 0 and negative == 0:
        return "no_effect"
    if positive > 0 and negative == 0:
        return "consistent_improvement"
    if negative > 0 and positive == 0:
        return "consistent_regression"
    return "mixed"


def _paired_effect_summary(items: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Summarize exact parent-to-candidate deltas for one case group."""
    if not items:
        return {
            "case_count": 0,
            "mean_delta_gain": 0.0,
            "median_delta_gain": 0.0,
            "positive_cases": 0,
            "negative_cases": 0,
            "zero_cases": 0,
            "win_rate": 0.0,
            "mean_delta_relative_player_mse": 0.0,
            "mean_player_mse_ratio": None,
            "median_player_mse_ratio": None,
            "component_delta_mean": {},
            "effect": "no_cases",
        }
    deltas = [float(item["delta_gain"]) for item in items]
    ratios = [
        float(item["player_mse_ratio"])
        for item in items
        if item.get("player_mse_ratio") is not None
    ]
    positive = sum(value > 0.0 for value in deltas)
    negative = sum(value < 0.0 for value in deltas)
    zero = len(deltas) - positive - negative
    component_names = sorted({
        str(name)
        for item in items
        for name in item.get("component_deltas", {})
    })
    component_delta_mean = {
        name: sum(float(item.get("component_deltas", {}).get(name, 0.0)) for item in items)
        / len(items)
        for name in component_names
    }
    return {
        "case_count": len(items),
        "baseline_mean_gain": sum(float(item["baseline_gain"]) for item in items) / len(items),
        "candidate_mean_gain": sum(float(item["candidate_gain"]) for item in items) / len(items),
        "mean_delta_gain": sum(deltas) / len(deltas),
        "median_delta_gain": float(statistics.median(deltas)),
        "positive_cases": positive,
        "negative_cases": negative,
        "zero_cases": zero,
        "win_rate": positive / len(deltas),
        "min_delta_gain": min(deltas),
        "max_delta_gain": max(deltas),
        "mean_delta_relative_player_mse": sum(
            float(item["delta_relative_player_mse"]) for item in items
        ) / len(items),
        "mean_player_mse_ratio": sum(ratios) / len(ratios) if ratios else None,
        "median_player_mse_ratio": float(statistics.median(ratios)) if ratios else None,
        "component_delta_mean": component_delta_mean,
        "effect": _effect_label(positive, negative, zero),
    }


def _paired_group_summary(
    items: Sequence[Mapping[str, Any]],
    key: str,
) -> dict[str, dict[str, Any]]:
    groups: dict[str, list[Mapping[str, Any]]] = {}
    for item in items:
        groups.setdefault(str(item.get(key, "")), []).append(item)
    return {
        name: _paired_effect_summary(group)
        for name, group in sorted(groups.items())
    }


def _paired_case(
    baseline: Mapping[str, Any],
    candidate: Mapping[str, Any],
    metadata: Mapping[str, Any],
    component_fields: Mapping[str, str],
) -> dict[str, Any]:
    baseline_mse = float(baseline.get("mse_player", 0.0))
    candidate_mse = float(candidate.get("mse_player", 0.0))
    ratio = candidate_mse / baseline_mse if baseline_mse > 0.0 else None
    component_deltas = {
        output_name: float(candidate[source_name]) - float(baseline[source_name])
        for output_name, source_name in component_fields.items()
        if source_name in baseline and source_name in candidate
    }
    return {
        **metadata,
        "baseline_gain": float(baseline.get("gain", 0.0)),
        "candidate_gain": float(candidate.get("gain", 0.0)),
        "delta_gain": float(candidate.get("gain", 0.0)) - float(baseline.get("gain", 0.0)),
        "baseline_relative_player_mse": float(baseline.get("relative_player_mse", 0.0)),
        "candidate_relative_player_mse": float(candidate.get("relative_player_mse", 0.0)),
        "delta_relative_player_mse": float(candidate.get("relative_player_mse", 0.0))
        - float(baseline.get("relative_player_mse", 0.0)),
        "player_mse_ratio": ratio,
        "component_deltas": component_deltas,
    }


def _timing_effect(
    baseline: Mapping[str, Any],
    candidate: Mapping[str, Any],
) -> dict[str, Any]:
    baseline_timing = baseline.get("timing", {})
    candidate_timing = candidate.get("timing", {})
    baseline_total = float(baseline_timing.get("api_total_seconds", 0.0))
    candidate_total = float(candidate_timing.get("api_total_seconds", 0.0))
    names = sorted(set(baseline_timing.get("api_seconds", {})) | set(candidate_timing.get("api_seconds", {})))
    by_api: dict[str, Any] = {}
    for name in names:
        baseline_seconds = float(baseline_timing.get("api_seconds", {}).get(name, 0.0))
        candidate_seconds = float(candidate_timing.get("api_seconds", {}).get(name, 0.0))
        by_api[name] = {
            "baseline_seconds": baseline_seconds,
            "candidate_seconds": candidate_seconds,
            "delta_seconds": candidate_seconds - baseline_seconds,
            "ratio": candidate_seconds / baseline_seconds if baseline_seconds > 0.0 else None,
            "baseline_calls": int(baseline_timing.get("api_calls", {}).get(name, 0)),
            "candidate_calls": int(candidate_timing.get("api_calls", {}).get(name, 0)),
        }
    return {
        "baseline_api_total_seconds": baseline_total,
        "candidate_api_total_seconds": candidate_total,
        "delta_api_total_seconds": candidate_total - baseline_total,
        "api_total_ratio": candidate_total / baseline_total if baseline_total > 0.0 else None,
        "by_api": by_api,
        "note": "same-host A/B diagnostic only; local seconds are not official runtime",
    }


def _paired_effect_diagnostics(
    baseline: Mapping[str, Any],
    candidate: Mapping[str, Any],
    focus_linear_roles: Sequence[str] = (),
) -> dict[str, Any]:
    """Explain whether one mechanism changed the exact same W/A cases.

    Unlike aggregate proxy means, this diagnostic pairs every case identity,
    compares parent and candidate output error, and keeps W/A or Q/K/V control
    arms as signed deltas.  It consumes saved evaluator results and invokes no
    candidate API, so one immutable parent JSON can be reused across trials.
    """
    if baseline.get("status") != "ok" or candidate.get("status") != "ok":
        return {"enabled": False, "reason": "baseline and candidate must both have status=ok"}

    baseline_linear = {
        _linear_case_identity(item): item
        for item in baseline.get("case_scores", {}).get("linear", [])
    }
    candidate_linear = {
        _linear_case_identity(item): item
        for item in candidate.get("case_scores", {}).get("linear", [])
    }
    baseline_attention = {
        _attention_case_identity(item): item
        for item in baseline.get("case_scores", {}).get("attention", [])
    }
    candidate_attention = {
        _attention_case_identity(item): item
        for item in candidate.get("case_scores", {}).get("attention", [])
    }
    if set(baseline_linear) != set(candidate_linear) or set(baseline_attention) != set(candidate_attention):
        return {
            "enabled": False,
            "reason": "case identities differ; baseline and candidate must use the same cache and panel",
            "baseline_linear_cases": len(baseline_linear),
            "candidate_linear_cases": len(candidate_linear),
            "baseline_attention_cases": len(baseline_attention),
            "candidate_attention_cases": len(candidate_attention),
        }

    standard_mismatches: list[dict[str, Any]] = []
    for domain, baseline_cases, candidate_cases in (
        ("linear", baseline_linear, candidate_linear),
        ("attention", baseline_attention, candidate_attention),
    ):
        for identity, baseline_item in baseline_cases.items():
            candidate_item = candidate_cases[identity]
            for field in ("mse_standard", "reference_energy"):
                left = float(baseline_item.get(field, 0.0))
                right = float(candidate_item.get(field, 0.0))
                if not math.isclose(left, right, rel_tol=1.0e-12, abs_tol=1.0e-15):
                    standard_mismatches.append({
                        "domain": domain,
                        "identity": list(identity),
                        "field": field,
                        "baseline": left,
                        "candidate": right,
                    })
                    break
    if standard_mismatches:
        return {
            "enabled": False,
            "reason": "standard/reference arms differ; saved results are not a valid paired panel",
            "mismatches": standard_mismatches[:16],
        }

    linear_component_fields = {
        "w_only_gain": "gain_w_only",
        "a_only_gain": "gain_a_only",
        "both_gain": "gain_both",
        "interaction_gain": "interaction_gain",
        "weight_operand_relative_mse": "weight_relative_mse",
        "activation_operand_relative_mse": "activation_relative_mse",
    }
    attention_component_fields = {
        "q_only_gain": "gain_q_only",
        "k_only_gain": "gain_k_only",
        "v_only_gain": "gain_v_only",
        "qk_only_gain": "gain_qk_only",
        "both_gain": "gain_both",
        "qk_interaction_gain": "qk_interaction_gain",
        "qkv_interaction_gain": "qkv_interaction_gain",
        "logit_mse": "logit_mse_player",
        "probability_mse": "probability_mse_player",
        "probability_kl": "probability_kl_player_to_reference",
    }
    linear_pairs: list[dict[str, Any]] = []
    for identity in sorted(baseline_linear):
        baseline_item = baseline_linear[identity]
        candidate_item = candidate_linear[identity]
        role = str(candidate_item.get("role", ""))
        linear_pairs.append(_paired_case(
            baseline_item,
            candidate_item,
            {
                "layer": int(candidate_item.get("layer", -1)),
                "role": role,
                "role_family": str(candidate_item.get("role_family", _linear_role_family(role))),
                "shape_bucket": str(candidate_item.get("shape_bucket", "")),
                "test_window": int(candidate_item.get("test_window", -1)),
                "test_split": str(candidate_item.get("test_split", "")),
                "test_length": int(candidate_item.get("test_length", -1)),
            },
            linear_component_fields,
        ))
    attention_pairs: list[dict[str, Any]] = []
    for identity in sorted(baseline_attention):
        baseline_item = baseline_attention[identity]
        candidate_item = candidate_attention[identity]
        attention_pairs.append(_paired_case(
            baseline_item,
            candidate_item,
            {
                "layer": int(candidate_item.get("layer", -1)),
                "test_window": int(candidate_item.get("test_window", -1)),
                "test_split": str(candidate_item.get("test_split", "")),
                "test_length": int(candidate_item.get("test_length", -1)),
            },
            attention_component_fields,
        ))

    selectors = tuple(dict.fromkeys(str(value).strip() for value in focus_linear_roles if str(value).strip()))
    def is_focus(item: Mapping[str, Any]) -> bool:
        return bool(
            item.get("role") in selectors
            or item.get("role_family") in selectors
        )

    focus_pairs = [item for item in linear_pairs if is_focus(item)]
    control_pairs = [item for item in linear_pairs if not is_focus(item)]
    if selectors and not focus_pairs:
        return {
            "enabled": False,
            "reason": "focus selectors matched no Linear role or role family",
            "focus_selectors": list(selectors),
            "available_roles": sorted({str(item.get("role", "")) for item in linear_pairs}),
            "available_role_families": sorted({
                str(item.get("role_family", "")) for item in linear_pairs
            }),
        }
    linear_payload = {
        "overall": _paired_effect_summary(linear_pairs),
        "focus_selectors": list(selectors),
        "focus": _paired_effect_summary(focus_pairs) if selectors else {"enabled": False, "reason": "no focus role/family supplied"},
        "control": _paired_effect_summary(control_pairs) if selectors else {"enabled": False, "reason": "no focus role/family supplied"},
        "by_role": _paired_group_summary(linear_pairs, "role"),
        "by_role_family": _paired_group_summary(linear_pairs, "role_family"),
        "by_layer": _paired_group_summary(linear_pairs, "layer"),
        "by_shape": _paired_group_summary(linear_pairs, "shape_bucket"),
        "by_split": _paired_group_summary(linear_pairs, "test_split"),
        "by_test_length": _paired_group_summary(linear_pairs, "test_length"),
        "worst_cases": sorted(linear_pairs, key=lambda item: float(item["delta_gain"]))[:16],
        "best_cases": sorted(linear_pairs, key=lambda item: float(item["delta_gain"]), reverse=True)[:16],
    }
    attention_payload = {
        "overall": _paired_effect_summary(attention_pairs),
        "by_layer": _paired_group_summary(attention_pairs, "layer"),
        "by_split": _paired_group_summary(attention_pairs, "test_split"),
        "by_test_length": _paired_group_summary(attention_pairs, "test_length"),
        "worst_cases": sorted(attention_pairs, key=lambda item: float(item["delta_gain"]))[:16],
        "best_cases": sorted(attention_pairs, key=lambda item: float(item["delta_gain"]), reverse=True)[:16],
    }
    return {
        "enabled": True,
        "baseline": str(baseline.get("candidate", "baseline")),
        "candidate": str(candidate.get("candidate", "candidate")),
        "formula": "paired delta gain = candidate gain - baseline gain on the same case; positive is better",
        "decision_policy": "descriptive only: inspect focus direction, sign consistency, control leakage, component deltas, worst cases, and same-host API timing; no fitted score or hidden threshold",
        "linear": linear_payload,
        "attention": attention_payload,
        "timing": _timing_effect(baseline, candidate),
        "score_delta": {
            key: float(candidate.get("score", {}).get(key, 0.0)) - float(baseline.get("score", {}).get(key, 0.0))
            for key in ("linear_mean", "attention_mean", "overall_mean")
        },
        "interpretation": {
            "linear_overall": linear_payload["overall"].get("effect"),
            "focus": linear_payload["focus"].get("effect") if selectors else "not_selected",
            "control": linear_payload["control"].get("effect") if selectors else "not_selected",
            "attention": attention_payload["overall"].get("effect"),
        },
    }


def _cpu_params(params: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {name: value.detach().to("cpu") for name, value in params.items()}


def evaluate_solution(
    path: Path,
    pack: PreparedPack,
    algorithm_device_name: str,
    decomposition: bool = True,
) -> dict[str, Any]:
    solution = load_solution(path)
    label = path.parent.name
    device = torch.device(algorithm_device_name)
    # Candidate APIs and evaluator control-arm matmuls use the same selected
    # device when CUDA is available.  The captured pack remains CPU-resident;
    # only the current case (and cached per-role weights) is moved, so the
    # diagnostics do not require duplicating the multi-GB cache on the GPU.
    score_device = device if device.type == "cuda" else torch.device("cpu")
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    wall_start = time.perf_counter()
    api_seconds = {name: 0.0 for name in REQUIRED_APIS}
    api_calls = {name: 0 for name in REQUIRED_APIS}
    weight_states: dict[tuple[int, str], tuple[Any, dict[str, torch.Tensor]]] = {}
    attention_states: dict[int, dict[str, Any]] = {}
    # Match the judge's state lifetime: calibration is paid once for every
    # layer/role (24*7 on Qwen) and once for every attention layer (24).  The
    # selected Linear/Attention cases below only vary the dynamic input.  A
    # fresh calibration for each case would silently turn the local evaluator
    # into a per-test oracle and reverses the meaning of the runtime budget.
    linear_calibration_indices = tuple(
        int(index) for index in pack.metadata.get("linear_calibration_indices", [0, 1])
    )
    if not linear_calibration_indices:
        raise RuntimeError("pack has no Linear calibration indices")
    print(f"[{label}] Linear calibration: {pack.layers * len(pack.roles)} layer/role states", flush=True)
    for layer in range(pack.layers):
        for role in pack.roles:
            weight_pair = _move_pair(pack.weights[layer][role], device)
            calibration = [
                _move_pair(
                    pack.linear_calibration_activations[role][sample][layer],
                    device,
                )
                for sample in linear_calibration_indices
            ]
            started = time.perf_counter()
            result = solution.hif4_calibration_and_quantize_weight(weight_pair[0], weight_pair[1], calibration)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            api_seconds["hif4_calibration_and_quantize_weight"] += time.perf_counter() - started
            api_calls["hif4_calibration_and_quantize_weight"] += 1
            if not isinstance(result, Mapping) or set(result) != {"weight_params", "activation_state"}:
                raise ValueError("weight calibration must return exactly weight_params and activation_state")
            validate_state(result["activation_state"])
            ref_weight_shape = dequantize_nvfp4(*pack.weights[layer][role]).shape
            validate_hif4_params(result["weight_params"], ref_weight_shape)
            weight_states[(layer, role)] = (result["activation_state"], _cpu_params(result["weight_params"]))
    attention_calibration_indices = tuple(range(len(pack.calibration_windows)))
    print(f"[{label}] Attention calibration: {pack.layers} layer states", flush=True)
    for layer in range(pack.layers):
        calibration = [
            _move_qkv(pack.calibration_qkv[sample][layer], device)
            for sample in attention_calibration_indices
        ]
        started = time.perf_counter()
        states = solution.hif4_calibration_attention(calibration, pack.q_heads, pack.kv_heads, pack.head_dim)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        api_seconds["hif4_calibration_attention"] += time.perf_counter() - started
        api_calls["hif4_calibration_attention"] += 1
        if not isinstance(states, Mapping) or set(states) != {"q_state", "k_state", "v_state"}:
            raise ValueError("attention calibration must return exactly q_state, k_state, v_state")
        for name in ("q_state", "k_state", "v_state"):
            validate_state(states[name])
        attention_states[layer] = dict(states)

    linear_details: list[dict[str, Any]] = []
    attention_details: list[dict[str, Any]] = []
    standard_weight_cache: dict[tuple[int, str], tuple[torch.Tensor, torch.Tensor]] = {}
    score_weight_cache: dict[tuple[int, str], tuple[torch.Tensor, torch.Tensor]] = {}
    print(f"[{label}] Linear scoring: {len(pack.linear_cases)} cases", flush=True)
    for case in pack.linear_cases:
        state, weight_params = weight_states[(case.layer, case.role)]
        activation_pair = _move_pair(pack.test_activations[case.role][case.test_window][case.layer], device)
        started = time.perf_counter()
        activation_params = solution.hif4_dynamic_quantize_activation(activation_pair[0], activation_pair[1], state)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        api_seconds["hif4_dynamic_quantize_activation"] += time.perf_counter() - started
        api_calls["hif4_dynamic_quantize_activation"] += 1
        ref_activation = dequantize_nvfp4(*pack.test_activations[case.role][case.test_window][case.layer]).to(torch.float32)
        cache_key = (case.layer, case.role)
        if cache_key not in standard_weight_cache:
            ref_weight = dequantize_nvfp4(*pack.weights[case.layer][case.role]).to(torch.float32)
            standard_weight_cache[cache_key] = (
                ref_weight,
                decode_standard_hif4(encode_standard_hif4(ref_weight)).to(torch.float32),
            )
        ref_weight, standard_weight = standard_weight_cache[cache_key]
        standard_activation = decode_standard_hif4(encode_standard_hif4(ref_activation)).to(torch.float32)
        player_activation = dequantize_hif4(_cpu_params(activation_params), ref_activation.shape).to(torch.float32)
        player_weight = dequantize_hif4(weight_params, ref_weight.shape).to(torch.float32)
        if cache_key not in score_weight_cache:
            score_weight_cache[cache_key] = (ref_weight.to(score_device), standard_weight.to(score_device))
        score_ref_weight, score_standard_weight = score_weight_cache[cache_key]
        score_ref_activation = ref_activation.to(score_device)
        score_standard_activation = standard_activation.to(score_device)
        score_player_activation = player_activation.to(score_device)
        score_player_weight = player_weight.to(score_device)
        reference = score_ref_activation @ score_ref_weight.T
        standard = score_standard_activation @ score_standard_weight.T
        player = score_player_activation @ score_player_weight.T
        if decomposition:
            weight_only = score_standard_activation @ score_player_weight.T
            activation_only = score_player_activation @ score_standard_weight.T
            details = _linear_error_source_details(
                standard,
                weight_only,
                activation_only,
                player,
                reference,
                ref_weight,
                standard_weight,
                player_weight,
                ref_activation,
                standard_activation,
                player_activation,
            )
        else:
            details = _score_details(standard, player, reference)
        details.update({
            "case_id": case.case_id,
            "layer": case.layer,
            "role": case.role,
            "role_family": _linear_role_family(case.role),
            "role_domain": "static_linear",
            "calibration_indices": list(case.calibration_indices),
            "test_window": case.test_window,
            "test_split": pack.test_windows[case.test_window].split,
            "test_length": len(pack.test_windows[case.test_window].input_ids),
            "input_width": int(ref_activation.shape[-1]),
            "output_width": int(ref_weight.shape[0]),
            "shape_bucket": (
                "hidden_to_hidden"
                if int(ref_activation.shape[-1]) == pack.hidden_size and int(ref_weight.shape[0]) == pack.hidden_size
                else "hidden_to_wide"
                if int(ref_activation.shape[-1]) == pack.hidden_size
                else "wide_to_hidden"
                if int(ref_weight.shape[0]) == pack.hidden_size
                else "other"
            ),
        })
        linear_details.append(details)
    print(f"[{label}] Attention scoring: {len(pack.attention_cases)} cases", flush=True)
    for case in pack.attention_cases:
        states = attention_states[case.layer]
        q_pair, k_pair, v_pair = (_move_pair(pair, device) for pair in pack.test_qkv[case.test_window][case.layer])
        started = time.perf_counter()
        q_params = solution.hif4_dynamic_quantize_q(q_pair[0], q_pair[1], pack.q_heads, pack.head_dim, states["q_state"])
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        api_seconds["hif4_dynamic_quantize_q"] += time.perf_counter() - started
        api_calls["hif4_dynamic_quantize_q"] += 1
        started = time.perf_counter()
        k_params = solution.hif4_dynamic_quantize_k(k_pair[0], k_pair[1], pack.kv_heads, pack.head_dim, states["k_state"])
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        api_seconds["hif4_dynamic_quantize_k"] += time.perf_counter() - started
        api_calls["hif4_dynamic_quantize_k"] += 1
        started = time.perf_counter()
        v_params = solution.hif4_dynamic_quantize_v(v_pair[0], v_pair[1], pack.kv_heads, pack.head_dim, states["v_state"])
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        api_seconds["hif4_dynamic_quantize_v"] += time.perf_counter() - started
        api_calls["hif4_dynamic_quantize_v"] += 1
        ref_q = dequantize_nvfp4(*pack.test_qkv[case.test_window][case.layer][0]).to(torch.float32)
        ref_k = dequantize_nvfp4(*pack.test_qkv[case.test_window][case.layer][1]).to(torch.float32)
        ref_v = dequantize_nvfp4(*pack.test_qkv[case.test_window][case.layer][2]).to(torch.float32)
        std_q = decode_standard_hif4(encode_standard_hif4(ref_q)).to(torch.float32)
        std_k = decode_standard_hif4(encode_standard_hif4(ref_k)).to(torch.float32)
        std_v = decode_standard_hif4(encode_standard_hif4(ref_v)).to(torch.float32)
        player_q = dequantize_hif4(_cpu_params(q_params), ref_q.shape).to(torch.float32)
        player_k = dequantize_hif4(_cpu_params(k_params), ref_k.shape).to(torch.float32)
        player_v = dequantize_hif4(_cpu_params(v_params), ref_v.shape).to(torch.float32)
        score_ref_q = ref_q.to(score_device)
        score_ref_k = ref_k.to(score_device)
        score_ref_v = ref_v.to(score_device)
        score_std_q = std_q.to(score_device)
        score_std_k = std_k.to(score_device)
        score_std_v = std_v.to(score_device)
        score_player_q = player_q.to(score_device)
        score_player_k = player_k.to(score_device)
        score_player_v = player_v.to(score_device)
        if decomposition:
            reference, reference_logits, reference_probabilities = _attention_trace(
                score_ref_q[None], score_ref_k[None], score_ref_v[None], pack.q_heads, pack.kv_heads, pack.head_dim
            )
            standard, standard_logits, standard_probabilities = _attention_trace(
                score_std_q[None], score_std_k[None], score_std_v[None], pack.q_heads, pack.kv_heads, pack.head_dim
            )
            player, player_logits, player_probabilities = _attention_trace(
                score_player_q[None], score_player_k[None], score_player_v[None], pack.q_heads, pack.kv_heads, pack.head_dim
            )
            q_only = _attention(
                score_player_q[None], score_std_k[None], score_std_v[None], pack.q_heads, pack.kv_heads, pack.head_dim
            )
            k_only = _attention(
                score_std_q[None], score_player_k[None], score_std_v[None], pack.q_heads, pack.kv_heads, pack.head_dim
            )
            v_only = _attention(
                score_std_q[None], score_std_k[None], score_player_v[None], pack.q_heads, pack.kv_heads, pack.head_dim
            )
            qk_only = _attention(
                score_player_q[None], score_player_k[None], score_std_v[None], pack.q_heads, pack.kv_heads, pack.head_dim
            )
            details = _attention_error_source_details(
                standard,
                q_only,
                k_only,
                v_only,
                qk_only,
                player,
                reference,
                reference_logits,
                standard_logits,
                player_logits,
                reference_probabilities,
                standard_probabilities,
                player_probabilities,
            )
        else:
            reference = _attention(
                score_ref_q[None], score_ref_k[None], score_ref_v[None], pack.q_heads, pack.kv_heads, pack.head_dim
            )
            standard = _attention(
                score_std_q[None], score_std_k[None], score_std_v[None], pack.q_heads, pack.kv_heads, pack.head_dim
            )
            player = _attention(
                score_player_q[None], score_player_k[None], score_player_v[None], pack.q_heads, pack.kv_heads, pack.head_dim
            )
            details = _score_details(standard, player, reference)
        details.update({
            "case_id": case.case_id,
            "layer": case.layer,
            "calibration_indices": list(case.calibration_indices),
            "test_window": case.test_window,
            "test_split": pack.test_windows[case.test_window].split,
            "test_length": int(ref_q.shape[0]),
        })
        attention_details.append(details)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    wall_seconds = time.perf_counter() - wall_start
    linear_scores = [float(item["gain"]) for item in linear_details]
    attention_scores = [float(item["gain"]) for item in attention_details]
    linear_sum = float(sum(linear_scores))
    attention_sum = float(sum(attention_scores))
    linear_mean = linear_sum / max(1, len(linear_details))
    attention_mean = attention_sum / max(1, len(attention_details))
    linear_by_role = {
        role: sum(item["gain"] for item in linear_details if item["role"] == role)
        / max(1, sum(item["role"] == role for item in linear_details))
        for role in pack.roles
    }
    linear_role_families = sorted({
        _linear_role_family(role) for role in pack.roles
    })
    linear_by_role_family = {
        family: sum(item["gain"] for item in linear_details if item["role_family"] == family)
        / max(1, sum(item["role_family"] == family for item in linear_details))
        for family in linear_role_families
    }
    attention_by_layer = {
        str(layer): sum(item["gain"] for item in attention_details if item["layer"] == layer)
        / max(1, sum(item["layer"] == layer for item in attention_details))
        for layer in sorted({item["layer"] for item in attention_details})
    }
    if decomposition:
        linear_decomposition: dict[str, Any] = {
            "enabled": True,
            "formula": {
                "arms": "E00=standard W+standard A; E10=candidate W+standard A; E01=standard W+candidate A; E11=candidate W+candidate A",
                "interaction_gain": "(E10+E01-E00-E11)/E00; positive means super-additive complementarity, negative means overlapping/diminishing returns",
            },
            "overall": _linear_decomposition_summary(linear_details),
            "by_role": _group_summary(linear_details, "role", _linear_decomposition_summary),
            "by_role_family": _group_summary(linear_details, "role_family", _linear_decomposition_summary),
            "by_layer": _group_summary(linear_details, "layer", _linear_decomposition_summary),
            "by_shape": _group_summary(linear_details, "shape_bucket", _linear_decomposition_summary),
            "by_test_length": _group_summary(linear_details, "test_length", _linear_decomposition_summary),
            "by_split": _group_summary(linear_details, "test_split", _linear_decomposition_summary),
        }
        attention_decomposition: dict[str, Any] = {
            "enabled": True,
            "formula": {
                "arms": "E000=standard Q/K/V; E100=candidate Q; E010=candidate K; E001=candidate V; E110=candidate Q+K; E111=candidate Q+K+V",
                "qk_interaction_gain": "(E100+E010-E000-E110)/E000; positive means super-additive complementarity",
                "qkv_interaction_gain": "(E110+E001-E000-E111)/E000; positive means super-additive complementarity",
            },
            "overall": _attention_decomposition_summary(attention_details),
            "by_layer": _group_summary(attention_details, "layer", _attention_decomposition_summary),
            "by_test_length": _group_summary(attention_details, "test_length", _attention_decomposition_summary),
            "by_split": _group_summary(attention_details, "test_split", _attention_decomposition_summary),
        }
    else:
        linear_decomposition = {"enabled": False}
        attention_decomposition = {"enabled": False}
    scope = _evaluation_scope(pack)
    return {
        "candidate": path.stem,
        "evaluation_scope": scope,
        "source": str(path.resolve()),
        "source_sha256": sha256_file(path),
        "score": {
            "linear_sum": linear_sum,
            "attention_sum": attention_sum,
            "total_sum": linear_sum + attention_sum,
            "linear_mean": linear_mean,
            "attention_mean": attention_mean,
            # This is the mean over the actual captured W/A cases.  No
            # Linear:Attention weighting is applied; compare runs only when
            # their case design and cache are identical.
            "overall_mean": (linear_sum + attention_sum)
            / max(1, len(linear_details) + len(attention_details)),
            "linear_role_macro_mean": sum(linear_by_role.values()) / max(1, len(pack.roles)),
            "attention_layer_macro_mean": sum(attention_by_layer.values()) / max(1, len(attention_by_layer)),
            "linear_cases": len(linear_details),
            "attention_cases": len(attention_details),
            "linear_by_role": linear_by_role,
            "linear_by_role_family": linear_by_role_family,
            "attention_by_layer": attention_by_layer,
        },
        "timing": {
            # The official rule times the six candidate quantization APIs.  Keep
            # that quantity separate from wall time, which also contains local
            # dequantization/MSE scoring and Python scheduling overhead.
            "api_seconds": api_seconds,
            "api_total_seconds": float(sum(api_seconds.values())),
            "api_calls": api_calls,
            "wall_seconds": wall_seconds,
            "local_api_under_300_indicator": float(sum(api_seconds.values())) < OFFICIAL_RUNTIME_LIMIT,
            "wall_under_300_indicator": wall_seconds < OFFICIAL_RUNTIME_LIMIT,
        },
        "case_scores": {
            "linear": linear_details,
            "attention": attention_details,
        },
        "decomposition": {
            "linear": linear_decomposition,
            "attention": attention_decomposition,
        },
        "diagnostic_config": {
            "error_source_decomposition": decomposition,
            "score_unchanged": True,
            "candidate_api_calls_unchanged": True,
        },
        "protocol": PROTOCOL,
    }


ARCHIVE_MANIFEST: dict[str, dict[str, Any]] = {
    "v001": {"path": "solutions/20260826_v001_current-baseline_score10250_time127s/solution.py", "official_score": 10250, "official_time": 127.0, "official_status": "pass", "official_cohort": "old-weight"},
    "v002": {"path": "solutions/20260826_v002_youxilee-hif4_score15000plus_timeNA/solution.py", "official_score": 15313, "official_time": 137.0, "official_status": "pass", "official_cohort": "old-weight"},
    "v013": {"path": "solutions/20260827_v013_c10-wide-activation-quadratic_score15799_time144s/solution.py", "official_score": 15799, "official_time": 144.0, "official_status": "pass", "official_cohort": "old-weight"},
    "v024": {"path": "solutions/20260827_v024_c21-gated-exact-cross-selection_score16043_time174s/solution.py", "official_score": 16043, "official_time": 173.8, "official_status": "pass", "official_cohort": "old-weight"},
    "v025": {"path": "solutions/20260827_v025_c21c-compliance-baseline/solution.py", "official_score": 14437, "official_time": 166.6, "official_status": "pass", "official_cohort": "old-weight"},
    "v030": {"path": "solutions/20260828_v030_c38-beam2-fullcov-official14092_time170.6s/solution.py", "official_score": 14092, "official_time": 170.57, "official_status": "pass", "official_cohort": "old-weight"},
    "v031": {"path": "solutions/20260828_v031_c39-fw-official21864_time161.3s/solution.py", "official_score": 21864, "official_time": 161.3, "official_status": "pass", "official_cohort": "old-weight"},
    "v032": {"path": "solutions/20260828_v032_c40-robust-blockldlq_official-score14432_time216.667s/solution.py", "official_score": 14432, "official_time": 216.667, "official_status": "pass", "official_cohort": "old-weight"},
    "v034": {"path": "solutions/20260829_v034_c41b-mha-k-center_scoreNA_timeNA/solution.py", "official_score": 21864, "official_time": 159.4, "official_status": "pass", "official_cohort": "old-weight"},
    "v051": {"path": "solutions/20260829_v051_c47b-grouping-threshold005_scoreNA_timeNA/solution.py", "official_score": 22451, "official_time": 234.0, "official_status": "pass", "official_cohort": "old-weight"},
    "v066": {"path": "solutions/20260829_v066_c66-activation-ratio100_scoreNA_timeNA/solution.py", "official_score": 22557, "official_time": 217.2, "official_status": "pass", "official_cohort": "old-weight"},
    "v072": {"path": "solutions/20260829_v072_c74-jdrq-hierarchy_scoreNA_timeNA/solution.py", "official_score": 22662, "official_time": 226.0, "official_status": "pass", "official_cohort": "old-weight"},
    "v074": {"path": "solutions/20260829_v074_c75-rowwise-jdrq_scoreNA_timeNA/solution.py", "official_score": 14561, "official_time": 188.9, "official_status": "pass", "official_cohort": "new-weight"},
    "v084": {"path": "solutions/20260830_v084_c84-gram64-sweep5_scoreNA_timeNA/solution.py", "official_score": 16517, "official_time": 252.563, "official_status": "pass", "official_cohort": "new-weight"},
    "v086": {"path": "solutions/20260830_v086_c86-attn-block-final_scoreNA_timeNA/solution.py", "official_score": 16744, "official_time": 222.7, "official_status": "pass", "official_cohort": "new-weight"},
    "v138": {"path": "solutions/20260901_v138_attention-static-v86-budget_scoreNA_timeNA/solution.py", "official_score": 15715, "official_time": 208.0, "official_status": "pass", "official_cohort": "new-weight"},
    "v139": {"path": "solutions/20260901_v139_linear-output-aware-gain_scoreNA_timeNA/solution.py", "official_score": 15716, "official_time": 202.0, "official_status": "pass", "official_cohort": "new-weight"},
    "v140": {"path": "solutions/20260901_v140_linear-roab-pair_rejected/solution.py", "official_score": 15838, "official_time": 207.0, "official_status": "pass", "official_cohort": "new-weight"},
    "v147": {"path": "solutions/20260901_v147_v86-attention-v140-linear_rejected/solution.py", "official_score": 16579, "official_time": 211.0, "official_status": "pass", "official_cohort": "new-weight"},
    "v098": {"path": "solutions/20260830_v098_b1-gqrb-margin-active_score293.793700_time406s/solution.py", "official_score": None, "official_time": None, "official_status": "timeout"},
    "v100": {"path": "solutions/20260830_v100_b2-pawv-diagonly-active_score293.797301_time392s/solution.py", "official_score": None, "official_time": None, "official_status": "wrong-answer/timeout"},
    "v107": {"path": "solutions/20260830_v107_l3-global-lrh-precision-parent_score295.157057_time481s/solution.py", "official_score": None, "official_time": None, "official_status": "wrong-answer"},
    "v121": {"path": "solutions/20260831_v121_c1b-structured-refresh2-accepted_score295.811281_time2180s/solution.py", "official_score": None, "official_time": None, "official_status": "timeout"},
}


def _load_eval_document(path: Path) -> dict[str, Any]:
    source = path.resolve()
    if not source.is_file():
        raise FileNotFoundError(f"evaluation JSON does not exist: {source}")
    value = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or not isinstance(value.get("results"), list):
        raise ValueError(f"evaluation JSON has no results list: {source}")
    if value.get("protocol") != PROTOCOL:
        raise ValueError(
            f"evaluation JSON protocol {value.get('protocol')!r} != {PROTOCOL!r}: {source}"
        )
    return value


def _select_eval_result(
    document: Mapping[str, Any],
    requested_name: str | None,
    label: str,
) -> Mapping[str, Any]:
    results = [item for item in document.get("results", []) if item.get("status") == "ok"]
    if requested_name:
        matches = [item for item in results if str(item.get("candidate", "")) == requested_name]
        if len(matches) != 1:
            raise ValueError(f"{label} result {requested_name!r} was not found uniquely")
        return matches[0]
    if len(results) != 1:
        raise ValueError(
            f"{label} JSON has {len(results)} successful results; select one explicitly"
        )
    return results[0]


def _format_optional(value: Any, digits: int = 6) -> str:
    if value is None:
        return "-"
    return f"{float(value):.{digits}f}"


def _write_paired_effect_report(path: Path, effect: Mapping[str, Any]) -> None:
    lines = [
        f"# Paired mechanism effect — {PROTOCOL}",
        "",
        f"- baseline: `{effect.get('baseline', '')}`",
        f"- candidate: `{effect.get('candidate', '')}`",
        "- evaluation scope: `paired-json-replay` (diagnosis only; never a proxy ranking score)",
        "- comparison: exact layer/role/window pairing; no candidate API is called during JSON replay",
        "- positive Δ gain means the candidate reduced output error on the same case",
    ]
    if not effect.get("enabled"):
        lines.extend(["", f"Pairing disabled: `{effect.get('reason', 'unknown')}`."])
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return

    linear = effect.get("linear", {})
    attention = effect.get("attention", {})
    rows: list[tuple[str, Mapping[str, Any]]] = [("Linear overall", linear.get("overall", {}))]
    if linear.get("focus", {}).get("enabled", True):
        rows.append((f"Linear focus:{','.join(linear.get('focus_selectors', []))}", linear.get("focus", {})))
        rows.append(("Linear control", linear.get("control", {})))
    rows.append(("Attention overall", attention.get("overall", {})))
    lines.extend([
        "",
        "## 效果总览",
        "",
        "| 范围 | cases | mean Δgain | median Δgain | 改善 | 回归 | 不变 | median MSE ratio | 结论 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ])
    for name, value in rows:
        lines.append(
            f"| {name} | {value.get('case_count', 0)} | {value.get('mean_delta_gain', 0.0):.6f} | "
            f"{value.get('median_delta_gain', 0.0):.6f} | {value.get('positive_cases', 0)} | "
            f"{value.get('negative_cases', 0)} | {value.get('zero_cases', 0)} | "
            f"{_format_optional(value.get('median_player_mse_ratio'))} | {value.get('effect', '')} |"
        )

    lines.extend([
        "",
        "## Linear role/family 配对差分",
        "",
        "| 分组 | cases | mean Δgain | median Δgain | 改善/回归/不变 | ΔW-only | ΔA-only | Δinteraction |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ])
    linear_groups: list[tuple[str, Mapping[str, Any]]] = []
    linear_groups.extend((f"family:{name}", value) for name, value in linear.get("by_role_family", {}).items())
    linear_groups.extend((f"role:{name}", value) for name, value in linear.get("by_role", {}).items())
    for name, value in linear_groups:
        components = value.get("component_delta_mean", {})
        lines.append(
            f"| {name} | {value.get('case_count', 0)} | {value.get('mean_delta_gain', 0.0):.6f} | "
            f"{value.get('median_delta_gain', 0.0):.6f} | "
            f"{value.get('positive_cases', 0)}/{value.get('negative_cases', 0)}/{value.get('zero_cases', 0)} | "
            f"{components.get('w_only_gain', 0.0):.6f} | {components.get('a_only_gain', 0.0):.6f} | "
            f"{components.get('interaction_gain', 0.0):.6f} |"
        )
    lines.extend([
        "",
        "W-only/A-only/interaction 都按各 case 的 standard-error 分母归一化；双侧强耦合时数值可能很大，"
        "应结合 Both、MSE ratio 和符号一起读取，不能相加解释为独立贡献。",
    ])

    lines.extend([
        "",
        "## 最坏 Linear 回归",
        "",
        "| layer | role | split | length | window | Δgain | MSE ratio |",
        "|---:|---|---|---:|---:|---:|---:|",
    ])
    for item in linear.get("worst_cases", [])[:8]:
        lines.append(
            f"| {item.get('layer', -1)} | {item.get('role', '')} | {item.get('test_split', '')} | "
            f"{item.get('test_length', -1)} | {item.get('test_window', -1)} | "
            f"{item.get('delta_gain', 0.0):.6f} | {_format_optional(item.get('player_mse_ratio'))} |"
        )

    attention_overall = attention.get("overall", {})
    attention_components = attention_overall.get("component_delta_mean", {})
    lines.extend([
        "",
        "## Attention 控制臂差分",
        "",
        "| ΔQ-only | ΔK-only | ΔV-only | ΔQK-only | ΔBoth | Δprobability MSE | Δprobability KL |",
        "|---:|---:|---:|---:|---:|---:|---:|",
        f"| {attention_components.get('q_only_gain', 0.0):.6f} | "
        f"{attention_components.get('k_only_gain', 0.0):.6f} | "
        f"{attention_components.get('v_only_gain', 0.0):.6f} | "
        f"{attention_components.get('qk_only_gain', 0.0):.6f} | "
        f"{attention_components.get('both_gain', 0.0):.6f} | "
        f"{attention_components.get('probability_mse', 0.0):.6e} | "
        f"{attention_components.get('probability_kl', 0.0):.6e} |",
        "",
        "## 同机 API 时间差分",
        "",
        "| API | baseline(s) | candidate(s) | Δ(s) | ratio | calls(base/candidate) |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    for name, value in effect.get("timing", {}).get("by_api", {}).items():
        lines.append(
            f"| {name} | {value.get('baseline_seconds', 0.0):.3f} | "
            f"{value.get('candidate_seconds', 0.0):.3f} | {value.get('delta_seconds', 0.0):.3f} | "
            f"{_format_optional(value.get('ratio'), 3)} | "
            f"{value.get('baseline_calls', 0)}/{value.get('candidate_calls', 0)} |"
        )
    lines.extend([
        "",
        "该报告只描述同 case 的机制效果和同机成本，不拟合官方分数，也不设置新的晋级阈值。",
    ])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _write_report(
    path: Path,
    result: Mapping[str, Any],
    pack: PreparedPack,
    paired_effect: Mapping[str, Any] | None = None,
) -> None:
    score = result["score"]
    timing = result["timing"]
    decomposition = result.get("decomposition", {})
    lines = [
        f"# {result['candidate']} — {PROTOCOL}",
        "",
        "本报告是本地 proxy；隐藏官方数据和鲲鹏硬件不可本地复制。",
        "",
        f"- evaluation scope: `{result.get('evaluation_scope', {}).get('kind', 'unknown')}` / `{result.get('evaluation_scope', {}).get('intent', 'do-not-compare')}`",
        f"- proxy ranking comparable: `{result.get('evaluation_scope', {}).get('comparable_for_proxy_ranking', False)}`; official-score equivalent: `False`",
        f"- calibration lengths: `{list(CALIBRATION_LENGTHS)}`",
        f"- cases: `{score['linear_cases']} Linear + {score['attention_cases']} Attention` (stratified real-W/A panel by default)",
        f"- calibration calls: `{timing['api_calls'].get('hif4_calibration_and_quantize_weight', 0)} weight + {timing['api_calls'].get('hif4_calibration_attention', 0)} attention` (shared state)",
        f"- input codec: `{pack.metadata.get('input_codec', NVFP4_INPUT_CODEC)}` / mode `{NVFP4_MODE}`",
        f"- test splits: `{pack.metadata.get('test_splits', [])}`",
        f"- source SHA256: `{result['source_sha256']}`",
        f"- data pack: `{pack.metadata.get('data_sha256', {})}`",
        "",
        "| 指标 | 值 |",
        "|---|---:|",
        f"| Linear mean | {score['linear_mean']:.9f} |",
        f"| Attention mean | {score['attention_mean']:.9f} |",
        f"| Overall mean (all captured cases) | {score['overall_mean']:.9f} |",
        f"| Linear role macro mean | {score['linear_role_macro_mean']:.9f} |",
        f"| Attention layer macro mean | {score['attention_layer_macro_mean']:.9f} |",
        f"| Candidate wall | {timing['wall_seconds']:.3f}s |",
        f"| Candidate API total | {timing['api_total_seconds']:.3f}s |",
    ]
    if paired_effect is not None:
        lines.extend(["", "## 父版本配对效果"])
        if paired_effect.get("enabled"):
            paired_linear = paired_effect.get("linear", {})
            paired_attention = paired_effect.get("attention", {})
            paired_rows: list[tuple[str, Mapping[str, Any]]] = [
                ("Linear overall", paired_linear.get("overall", {})),
            ]
            if paired_linear.get("focus", {}).get("enabled", True):
                paired_rows.extend([
                    (f"Linear focus:{','.join(paired_linear.get('focus_selectors', []))}", paired_linear.get("focus", {})),
                    ("Linear control", paired_linear.get("control", {})),
                ])
            paired_rows.append(("Attention overall", paired_attention.get("overall", {})))
            lines.extend([
                "",
                f"基线：`{paired_effect.get('baseline', '')}`；候选：`{paired_effect.get('candidate', '')}`。",
                "",
                "| 范围 | cases | mean Δgain | median Δgain | 改善/回归/不变 | median MSE ratio | 结论 |",
                "|---|---:|---:|---:|---:|---:|---|",
            ])
            for name, value in paired_rows:
                lines.append(
                    f"| {name} | {value.get('case_count', 0)} | {value.get('mean_delta_gain', 0.0):.6f} | "
                    f"{value.get('median_delta_gain', 0.0):.6f} | "
                    f"{value.get('positive_cases', 0)}/{value.get('negative_cases', 0)}/{value.get('zero_cases', 0)} | "
                    f"{_format_optional(value.get('median_player_mse_ratio'))} | {value.get('effect', '')} |"
                )
            lines.extend([
                "",
                "完整 role/family、W/A 来源、最坏 case、Attention 控制臂和 API 时间差分位于 JSON `paired_effect`。",
            ])
        else:
            lines.extend(["", f"配对已禁用：`{paired_effect.get('reason', 'unknown')}`。"])
    lines.extend([
        "",
        "## 误差源分解（evaluator-only）",
        "",
        "控制臂只在评测器内重算，不增加六个候选 API 调用，也不改变主分数。gain 为相对标准输出误差的改善；interaction 为正表示超加性互补，负表示收益重叠或递减。",
    ])
    linear_decomposition = decomposition.get("linear", {})
    if linear_decomposition.get("enabled"):
        lines.extend([
            "",
            "### Linear：W / A / 交互",
            "",
            "| 分组 | cases | W-only gain | A-only gain | Both gain | interaction | W operand rel-MSE | A operand rel-MSE |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ])
        linear_rows = [("overall", linear_decomposition.get("overall", {}))]
        linear_rows.extend(
            (f"family:{family}", value)
            for family, value in linear_decomposition.get("by_role_family", {}).items()
        )
        linear_rows.extend(
            (f"role:{role}", value)
            for role, value in linear_decomposition.get("by_role", {}).items()
        )
        linear_rows.extend(
            (f"shape:{shape}", value)
            for shape, value in linear_decomposition.get("by_shape", {}).items()
        )
        for name, value in linear_rows:
            gains = value.get("gain", {})
            operands = value.get("operand_relative_mse", {})
            lines.append(
                f"| {name} | {value.get('case_count', 0)} | {gains.get('w_only', 0.0):.6f} | "
                f"{gains.get('a_only', 0.0):.6f} | {gains.get('both', 0.0):.6f} | "
                f"{gains.get('interaction', 0.0):.6f} | {operands.get('weight', 0.0):.6e} | "
                f"{operands.get('activation', 0.0):.6e} |"
            )
        lines.extend([
            "",
            f"Linear overall interpretation: `{linear_decomposition.get('overall', {}).get('interpretation', 'unknown')}`.",
            "Linear 的完整 layer/role/window 结果位于 JSON 的 `decomposition.linear` 和 `case_scores.linear`；优先查看 role 与 shape 行，再定位对应 layer/case。",
        ])
    else:
        lines.extend(["", "Linear 分解：已通过 `--no-decomposition` 关闭。"])

    attention_decomposition = decomposition.get("attention", {})
    if attention_decomposition.get("enabled"):
        lines.extend([
            "",
            "### Attention：Q / K / V / softmax",
            "",
            "| 分组 | cases | Q-only | K-only | V-only | QK-only | Both | QK interaction | QKV interaction |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ])
        attention_rows = [("overall", attention_decomposition.get("overall", {}))]
        attention_rows.extend(
            (f"layer:{layer}", value)
            for layer, value in attention_decomposition.get("by_layer", {}).items()
        )
        attention_rows.extend(
            (f"length:{length}", value)
            for length, value in attention_decomposition.get("by_test_length", {}).items()
        )
        for name, value in attention_rows:
            gains = value.get("gain", {})
            lines.append(
                f"| {name} | {value.get('case_count', 0)} | {gains.get('q_only', 0.0):.6f} | "
                f"{gains.get('k_only', 0.0):.6f} | {gains.get('v_only', 0.0):.6f} | "
                f"{gains.get('qk_only', 0.0):.6f} | {gains.get('both', 0.0):.6f} | "
                f"{gains.get('qk_interaction', 0.0):.6f} | {gains.get('qkv_interaction', 0.0):.6f} |"
            )
        intermediate = attention_decomposition.get("overall", {}).get("intermediate", {})
        lines.extend([
            "",
            "| Attention 中间量（overall） | standard | player |",
            "|---|---:|---:|",
            f"| logit MSE vs reference | {intermediate.get('logit_mse_standard', 0.0):.6e} | {intermediate.get('logit_mse_player', 0.0):.6e} |",
            f"| probability MSE vs reference | {intermediate.get('probability_mse_standard', 0.0):.6e} | {intermediate.get('probability_mse_player', 0.0):.6e} |",
            f"| probability KL(reference || estimate) | {intermediate.get('probability_kl_standard_to_reference', 0.0):.6e} | {intermediate.get('probability_kl_player_to_reference', 0.0):.6e} |",
            "",
            f"Attention overall interpretation: `{attention_decomposition.get('overall', {}).get('interpretation', 'unknown')}`.",
            "Attention 的完整 layer/length/window 结果位于 JSON 的 `decomposition.attention` 和 `case_scores.attention`。",
        ])
    else:
        lines.extend(["", "Attention 分解：已通过 `--no-decomposition` 关闭。"])

    lines.extend([
        "",
        "官方成绩只保留为独立历史字段，不参与本地 proxy 评分或时间换算。",
    ])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _trend_diagnostics(results: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Compare local ordering with known official ordering, without fitting it.

    This is intentionally a diagnostic rather than a score correction.  A
    local proxy that disagrees with the same-cohort official anchors must be
    treated as a failed proxy and cannot be used to promote an algorithm.
    Historical scoring revisions are kept in separate cohorts.
    """
    eligible: list[tuple[str, str, float, float]] = []
    for result in results:
        if result.get("status") != "ok":
            continue
        official = result.get("official", {})
        score = official.get("score")
        cohort = official.get("cohort")
        local = result.get("score", {}).get("overall_mean")
        if cohort is None or score is None or local is None:
            continue
        try:
            official_value = float(score)
            local_value = float(local)
        except (TypeError, ValueError):
            continue
        if not (math.isfinite(official_value) and math.isfinite(local_value)):
            continue
        eligible.append((str(result.get("candidate", "?")), str(cohort), local_value, official_value))

    pairs: list[dict[str, Any]] = []
    concordant = 0
    inverted = 0
    tied = 0
    for index, left in enumerate(eligible):
        for right in eligible[index + 1:]:
            if left[1] != right[1]:
                continue
            local_delta = left[2] - right[2]
            official_delta = left[3] - right[3]
            if local_delta == 0.0 or official_delta == 0.0:
                tied += 1
                relation = "tie"
            elif local_delta * official_delta > 0.0:
                concordant += 1
                relation = "concordant"
            else:
                inverted += 1
                relation = "inversion"
            if relation == "inversion":
                pairs.append({
                    "left": left[0],
                    "right": right[0],
                    "local_delta": local_delta,
                    "official_delta": official_delta,
                    "cohort": left[1],
                })
    compared = concordant + inverted + tied
    return {
        "eligible_candidates": [item[0] for item in eligible],
        "cohort": sorted({item[1] for item in eligible}),
        "pair_count": compared,
        "concordant_pairs": concordant,
        "inverted_pairs": inverted,
        "tied_pairs": tied,
        "status": "pass" if inverted == 0 and compared > 0 else ("inversion_detected" if inverted else "insufficient_anchors"),
        "inversions": pairs[:32],
        "note": "diagnostic only; official scores are never used to alter proxy means",
    }


def _archive_output(
    data_source: str,
    capture_seconds: float,
    cache_path: Path,
    prepared: PreparedPack,
    results: Sequence[Mapping[str, Any]],
    paired_effect: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    scope = _evaluation_scope(prepared)
    return {
        "protocol": PROTOCOL,
        "evaluation_scope": scope,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "data_source": data_source,
        "capture_seconds": capture_seconds,
        "cache": str(cache_path),
        "protocol_config": {
            "model": MODEL_NAME,
            "linear_roles": list(prepared.roles),
            "calibration_lengths": list(CALIBRATION_LENGTHS),
            "test_length": TEST_LENGTH,
            "test_lengths": list(TEST_LENGTHS),
            "test_windows": TEST_WINDOW_COUNT,
            "linear_cases": len(prepared.linear_cases),
            "attention_cases": len(prepared.attention_cases),
            "case_design_scope": "stratified all-layer/role real-W/A panel by default; --effect-panel selects depth-spread paired iteration cases; --full-cases enables Cartesian stress; optional CLI limits are prefix smoke-only",
            "case_design": prepared.metadata.get("case_design", DEFAULT_CASE_DESIGN),
            "panel_window_indices": prepared.metadata.get("panel_window_indices", list(PANEL_WINDOW_INDICES)),
            "calibration_call_graph": prepared.metadata.get(
                "calibration_call_graph", "all-layer-role-once; attention-layer-once"
            ),
            "input_codec": NVFP4_INPUT_CODEC,
            "runtime_limit_seconds": OFFICIAL_RUNTIME_LIMIT,
            "score_formula": "proxy gain=(MSE_STD-MSE_PLAYER)/MSE_STD per case; means are unweighted over actual captured cases",
            "runtime_measurement": "sum of elapsed six-API calls; wall_seconds is reported separately and neither is official time",
            "error_source_decomposition": "candidate API outputs are reused in evaluator-only Linear W/A and Attention Q/K/V control arms; it does not alter proxy means or API call counts",
            "trend_validation": "same-cohort pairwise ordering against user-confirmed anchors; diagnostic-only",
            "paired_effect": "reuse an immutable parent JSON and compare exact layer/role/window output errors; report focus/control signs and W/A or Q/K/V source deltas",
            "scope_contract": "only default-panel runs are comparable for local proxy ranking; effect-panel is paired-only; full-cartesian is stress-only; explicit limits are smoke-only; none are official-score equivalents",
        },
        "data_metadata": prepared.metadata,
        "trend_diagnostics": _trend_diagnostics(results),
        "linear_candidate_role_diagnostics": _linear_candidate_role_diagnostics(results),
        "paired_effect": dict(paired_effect) if paired_effect is not None else None,
        "results": list(results),
    }


def _write_archive_report(
    path: Path,
    data_source: str,
    capture_seconds: float,
    results: Sequence[Mapping[str, Any]],
    prepared: PreparedPack | None = None,
) -> None:
    trend = _trend_diagnostics(results)
    role_diagnostics = _linear_candidate_role_diagnostics(results)
    linear_count = len(prepared.linear_cases) if prepared is not None else (
        len(results[0].get("case_scores", {}).get("linear", [])) if results else 0
    )
    attention_count = len(prepared.attention_cases) if prepared is not None else (
        len(results[0].get("case_scores", {}).get("attention", [])) if results else 0
    )
    lines = [
        f"# {PROTOCOL} archive evaluation",
        "",
        f"- data source: `{data_source}`",
        f"- evaluation scopes: `{sorted({result.get('evaluation_scope', {}).get('kind', 'unknown') for result in results})}`",
        "- proxy ranking is valid only for identical `default-panel` cache/panel runs; effect-panel, full-stress, smoke-prefix and external probes are not ranking scores",
        f"- capture seconds: `{capture_seconds:.3f}`",
        f"- calibration lengths: `{list(CALIBRATION_LENGTHS)}`",
        f"- case counts: `{linear_count} Linear + {attention_count} Attention` (stratified real-W/A panel by default)",
        f"- official trend audit: `{trend['status']}` ({trend['concordant_pairs']} concordant / {trend['inverted_pairs']} inverted / {trend['tied_pairs']} tied pairs)",
        "- trend audit is a same-cohort diagnostic only; it never changes a proxy score",
        "- error-source decomposition is stored per candidate in JSON `decomposition`/`case_scores`; archive table remains score-only",
        "",
        "| Candidate | Status | Linear mean | Attention mean | Overall mean | API total(s) | API calls | Wall(s) | Official status |",
        "|---|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for result in results:
        score = result.get("score", {})
        timing = result.get("timing", {})
        if result.get("status") == "ok":
            linear_mean = f"{score['linear_mean']:.6f}"
            attention_mean = f"{score['attention_mean']:.6f}"
            overall_mean = f"{score['overall_mean']:.6f}"
            api_total = f"{timing['api_total_seconds']:.3f}"
            api_calls = str(sum(timing.get('api_calls', {}).values()))
            wall = f"{timing['wall_seconds']:.3f}"
        else:
            linear_mean = attention_mean = overall_mean = api_total = api_calls = wall = "-"
        lines.append(
            f"| {result['candidate']} | {result['status']} | "
            f"{linear_mean} | {attention_mean} | {overall_mean} | {api_total} | {api_calls} | {wall} | "
            f"{result.get('official', {}).get('status', '')} |"
        )
    diagnostic_results = [
        result for result in results
        if result.get("status") == "ok"
        and result.get("decomposition", {}).get("linear", {}).get("enabled")
        and result.get("decomposition", {}).get("attention", {}).get("enabled")
    ]
    if diagnostic_results:
        lines.extend([
            "",
            "## Error-source overall (evaluator-only)",
            "",
            "| Candidate | Linear W-only | Linear A-only | Linear Both | Linear interaction | Attention QK-only | Attention V-only | Attention Both |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ])
        for result in diagnostic_results:
            linear = result["decomposition"]["linear"]["overall"]["gain"]
            attention = result["decomposition"]["attention"]["overall"]["gain"]
            lines.append(
                f"| {result['candidate']} | {linear['w_only']:.6f} | {linear['a_only']:.6f} | "
                f"{linear['both']:.6f} | {linear['interaction']:.6f} | {attention['qk_only']:.6f} | "
                f"{attention['v_only']:.6f} | {attention['both']:.6f} |"
            )
        lines.extend([
            "",
            "逐 role/layer/shape/length 细分仍位于各候选 JSON 的 `decomposition` 字段。",
        ])
    if role_diagnostics.get("enabled"):
        lines.extend([
            "",
            "## 跨候选静态 Linear role 差分（evaluator-only）",
            "",
            f"基线：`{role_diagnostics['baseline']}`；Δ gain = candidate gain − baseline gain，正值表示同一 layer/role/window 变好。",
            "",
            "| 候选 | 分组 | cases | mean Δ gain | 正向 | 负向 | 最差 Δ | 最好 Δ |",
            "|---|---|---:|---:|---:|---:|---:|---:|",
        ])
        for candidate, payload in role_diagnostics.get("candidates", {}).items():
            rows: list[tuple[str, Mapping[str, Any]]] = []
            rows.extend((f"family:{name}", value) for name, value in payload.get("by_role_family", {}).items())
            rows.extend((f"role:{name}", value) for name, value in payload.get("by_role", {}).items())
            for group_name, value in rows:
                lines.append(
                    f"| {candidate} | {group_name} | {value.get('case_count', 0)} | "
                    f"{value.get('mean_delta_gain', 0.0):.6f} | {value.get('positive_cases', 0)} | "
                    f"{value.get('negative_cases', 0)} | {value.get('min_delta_gain', 0.0):.6f} | "
                    f"{value.get('max_delta_gain', 0.0):.6f} |"
                )
        lines.extend([
            "",
            "最差 layer/role/window 已写入 JSON `linear_candidate_role_diagnostics.*.worst_cases`；该差分只用于定位回归，不改变任何候选分数。",
        ])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_run_report(
    path: Path,
    data_source: str,
    capture_seconds: float,
    results: Sequence[Mapping[str, Any]],
    prepared: PreparedPack,
    archive: bool,
    paired_effect: Mapping[str, Any] | None = None,
) -> None:
    """Write detailed diagnostics for a single run and compact archive output otherwise."""
    if not archive and len(results) == 1 and results[0].get("status") == "ok":
        _write_report(path, results[0], prepared, paired_effect=paired_effect)
    else:
        _write_archive_report(path, data_source, capture_seconds, results, prepared)


def _focus_selectors(value: str | None) -> tuple[str, ...]:
    if not value:
        return ()
    return tuple(dict.fromkeys(item.strip() for item in value.split(",") if item.strip()))


def _run_saved_comparison(args: argparse.Namespace) -> dict[str, Any]:
    if args.baseline_json is None:
        raise ValueError("--candidate-json requires --baseline-json")
    baseline_document = _load_eval_document(args.baseline_json)
    candidate_document = _load_eval_document(args.candidate_json)
    baseline_result = _select_eval_result(
        baseline_document, args.baseline_result, "baseline"
    )
    candidate_result = _select_eval_result(
        candidate_document, args.candidate_result, "candidate"
    )
    paired_effect = _paired_effect_diagnostics(
        baseline_result,
        candidate_result,
        _focus_selectors(args.focus_linear_roles),
    )
    output = {
        "protocol": PROTOCOL,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "mode": "paired-json-replay",
        "evaluation_scope": {
            "kind": "paired-json-replay",
            "intent": "paired-mechanism-diagnosis",
            "comparable_for_proxy_ranking": False,
            "paired_only": True,
            "stress_only": False,
            "smoke_only": False,
            "official_score_equivalent": False,
        },
        "baseline_json": str(args.baseline_json.resolve()),
        "candidate_json": str(args.candidate_json.resolve()),
        "paired_effect": paired_effect,
        "results": [candidate_result],
    }
    _write_json(args.output, output)
    if args.report:
        _write_paired_effect_report(args.report, paired_effect)
    return output


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.candidate_json is not None:
        if args.solution is not None or args.archive:
            raise ValueError("--candidate-json cannot be combined with --solution or --archive")
        return _run_saved_comparison(args)
    if args.baseline_json is not None and (args.solution is None or args.archive):
        raise ValueError("--baseline-json requires one --solution run or --candidate-json replay")
    baseline_result: Mapping[str, Any] | None = None
    if args.baseline_json is not None:
        baseline_document = _load_eval_document(args.baseline_json)
        baseline_result = _select_eval_result(
            baseline_document, args.baseline_result, "baseline"
        )
    focus_roles = _focus_selectors(args.focus_linear_roles)
    cache_path = args.cache.resolve()
    capture_started = time.perf_counter()
    if args.cache_mode == "read":
        raw = load_pack(cache_path)
        data_source = "cache"
    else:
        raw = capture_pack(args.capture_device)
        data_source = "model_forward"
        if args.cache_mode in {"auto", "write"}:
            save_pack(raw, cache_path)
    prepared = prepare_pack(
        raw,
        linear_count=args.linear_cases,
        attention_count=args.attention_cases,
        full_cases=args.full_cases,
        effect_panel=args.effect_panel,
    )
    capture_seconds = time.perf_counter() - capture_started
    if args.archive:
        candidates = [(name, item) for name, item in ARCHIVE_MANIFEST.items()]
    else:
        if args.solution is None:
            # ``--cache-mode write`` is a supported capture-only operation.
            # The old evaluator captured a multi-gigabyte pack successfully
            # and then discarded it with this late argument error.
            output = _archive_output(data_source, capture_seconds, cache_path, prepared, [])
            _write_json(args.output, output)
            if args.report:
                _write_archive_report(args.report, data_source, capture_seconds, [], prepared)
            print(f"[capture] wrote {cache_path}", flush=True)
            return output
        anchor = OFFICIAL_TREND_ANCHORS.get(args.name)
        candidates = [(
            args.name,
            {
                "path": str(args.solution),
                "official_score": None if anchor is None else anchor["score"],
                "official_time": None if anchor is None else anchor["time_seconds"],
                "official_status": "unregistered" if anchor is None else anchor["status"],
                "official_cohort": None if anchor is None else anchor["cohort"],
            },
        )]
    results: list[dict[str, Any]] = []
    paired_effect: Mapping[str, Any] | None = None
    for name, item in candidates:
        source = (ROOT / item["path"]).resolve()
        print(f"[{name}] evaluating {source}", flush=True)
        try:
            result = evaluate_solution(
                source,
                prepared,
                args.algorithm_device,
                decomposition=args.decomposition,
            )
            result["candidate"] = name
            result["official"] = {
                "score": item.get("official_score"),
                "time_seconds": item.get("official_time"),
                "status": item.get("official_status"),
                "cohort": item.get("official_cohort"),
            }
            result["status"] = "ok"
        except Exception as exc:
            result = {
                "candidate": name,
                "source": str(source),
                "source_sha256": sha256_file(source) if source.is_file() else None,
                "official": {"score": item.get("official_score"), "time_seconds": item.get("official_time"), "status": item.get("official_status"), "cohort": item.get("official_cohort")},
                "status": "error",
                "error": f"{type(exc).__name__}: {exc}",
            }
            print(f"[{name}] ERROR {result['error']}", flush=True)
        results.append(result)
        if baseline_result is not None and len(results) == 1:
            paired_effect = _paired_effect_diagnostics(
                baseline_result, result, focus_roles
            )
        # Each archive solution is an independent submission.  Do not let a
        # module-level tensor/cache from one candidate change the next
        # candidate's CUDA memory budget or its result.
        for module_name in list(sys.modules):
            if module_name.startswith("_hif4_official_"):
                sys.modules.pop(module_name, None)
        gc.collect()
        if args.algorithm_device.startswith("cuda") and torch.cuda.is_available():
            torch.cuda.empty_cache()
        # Checkpoint after every candidate so a long archive run remains
        # inspectable if a process is interrupted.
        _write_json(
            args.output,
            _archive_output(
                data_source,
                capture_seconds,
                cache_path,
                prepared,
                results,
                paired_effect=paired_effect,
            ),
        )
        if args.report:
            _write_run_report(
                args.report,
                data_source,
                capture_seconds,
                results,
                prepared,
                args.archive,
                paired_effect=paired_effect,
            )
    output = _archive_output(
        data_source,
        capture_seconds,
        cache_path,
        prepared,
        results,
        paired_effect=paired_effect,
    )
    _write_json(args.output, output)
    if args.report:
        _write_run_report(
            args.report,
            data_source,
            capture_seconds,
            results,
            prepared,
            args.archive,
            paired_effect=paired_effect,
        )
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", action="store_true", help="evaluate every archived version with an official result")
    parser.add_argument("--solution", type=Path, help="one solution.py when --archive is not used")
    parser.add_argument("--name", default="candidate", help="name for --solution")
    parser.add_argument(
        "--baseline-json",
        type=Path,
        help="reuse one saved proxy-v2 parent result for exact paired mechanism deltas",
    )
    parser.add_argument(
        "--candidate-json",
        type=Path,
        help="compare a saved candidate JSON to --baseline-json without running any candidate API",
    )
    parser.add_argument("--baseline-result", help="candidate name to select from a multi-result baseline JSON")
    parser.add_argument("--candidate-result", help="candidate name to select from a multi-result candidate JSON")
    parser.add_argument(
        "--focus-linear-roles",
        default="",
        help="comma-separated static roles or families (for example fc or fc_gate,fc_up) highlighted against unchanged controls",
    )
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--cache-mode", choices=("auto", "read", "write", "off"), default="auto")
    parser.add_argument(
        "--linear-cases",
        type=int,
        default=LINEAR_CASE_COUNT,
        help="optional Linear case prefix limit for smoke only; default uses 168-case stratified real-W/A panel",
    )
    parser.add_argument(
        "--attention-cases",
        type=int,
        default=ATTENTION_CASE_COUNT,
        help="optional Attention case prefix limit for smoke only; default uses 120-case five-length panel",
    )
    parser.add_argument(
        "--full-cases",
        action="store_true",
        help="expand every captured layer/role/window tuple for stress only; default uses the stratified real-W/A panel",
    )
    parser.add_argument(
        "--effect-panel",
        action="store_true",
        help="iteration panel: eight depth-spread layers x all Linear roles plus five depth/length Attention sentinels",
    )
    parser.add_argument("--capture-device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--algorithm-device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--no-decomposition",
        dest="decomposition",
        action="store_false",
        help="disable evaluator-only W/A and Q/K/V error-source diagnostics for a fast smoke run",
    )
    parser.set_defaults(decomposition=True)
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts" / "official_eval" / "archive.json")
    parser.add_argument("--report", type=Path, default=ROOT / "logs" / "official_eval" / "archive.md")
    return parser


if __name__ == "__main__":
    output = run(build_parser().parse_args())
    paired = output.get("paired_effect")
    success = (
        bool(output.get("results"))
        and all(item.get("status") == "ok" for item in output["results"])
        and (paired is None or paired.get("enabled") is True)
    )
    raise SystemExit(0 if success else 1)
