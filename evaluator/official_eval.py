"""Canonical local HiF4 proxy evaluator.

``proxy-v2`` is deliberately a *local trend* evaluator, not a claim that the
hidden judge can be reproduced.  It preserves the published per-case score
formula and API contract while fixing the old proxy's two largest biases:

* the default panel enumerates every captured layer/role/window tensor (no
  Linear:Attention weighting or prefix sampling);
* calibration state lifetime follows the judge call graph: one state per
  layer/role (Linear) and one per layer (Attention), while cases only vary the
  dynamic inputs.  This prevents a per-case calibration oracle.

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
# No artificial Linear:Attention weighting is applied.  By default the case
# design expands to every captured W/A tensor: 24*7*windows for Linear and
# 24*windows for Attention on Qwen.  The CLI limits below are opt-in smoke
# overrides only; they are recorded and must not be used for ranking.
LINEAR_CASE_COUNT: int | None = None
ATTENTION_CASE_COUNT: int | None = None
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
    # Reject stale snapshots produced by the pre-v1 transpose bug.  The public
    # API receives native [out_features, in_features] weights.
    if not isinstance(weights, list) or not weights:
        raise RuntimeError("official data pack has no layer weights")
    if len(weights) != layers:
        raise RuntimeError(f"data pack declares {layers} layers but stores {len(weights)} weight layers")
    expected_kv_width = kv_heads * head_dim
    for layer_index, per_layer in enumerate(weights):
        for role in ROLES:
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
        for role in {"fc_gate", "fc_up", "proj"}:
            value = per_layer[role]
            rows, cols = map(int, value.shape)
            if role in {"fc_gate", "fc_up"} and not (rows > cols and cols == hidden_size):
                raise RuntimeError(f"layer {layer_index} weight {role} is not native out-in layout")
            if role == "proj" and not (rows == hidden_size and cols > rows):
                raise RuntimeError(f"layer {layer_index} weight proj is not native out-in layout")
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
        for role in ROLES:
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
        kv_heads, head_dim, metadata,
    )


def _pair(value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    carrier, scale = nvfp4_encode(value, NVFP4_MODE)
    return carrier.contiguous(), scale.contiguous()


def _choose_cases(
    pack: RawPack,
    linear_count: int | None = LINEAR_CASE_COUNT,
    attention_count: int | None = ATTENTION_CASE_COUNT,
) -> tuple[list[LinearCase], list[AttentionCase]]:
    """Enumerate real captured tensors; optional limits are smoke-only.

    The default is the Cartesian product of every layer, role and holdout
    window.  This makes the local score a property of the captured model
    tensors rather than a hash-prefix sample.  ``linear_count`` and
    ``attention_count`` are explicit development overrides and are never
    described as a ranking panel.
    """
    if len(pack.test_windows) == 0 or len(pack.calibration_windows) == 0:
        raise RuntimeError("case pool has no calibration/test windows")
    linear_calibration_indices = tuple(range(min(2, len(pack.calibration_windows))))
    if not linear_calibration_indices:
        raise RuntimeError("case pool has no Linear calibration windows")
    linear_pool = [
        (layer, role, test_window)
        for test_window in range(len(pack.test_windows))
        for layer in range(pack.layers)
        for role in ROLES
    ]
    attention_pool = [
        (layer, test_window)
        for test_window in range(len(pack.test_windows))
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
    cal_act = {role: [[_pair(raw.calibration_activations[role][sample][layer]) for layer in range(raw.layers)] for sample in range(len(raw.calibration_windows))] for role in ROLES}
    test_act = {role: [[_pair(raw.test_activations[role][sample][layer]) for layer in range(raw.layers)] for sample in range(len(raw.test_windows))] for role in ROLES}
    cal_qkv = [[{"q": _pair(q), "k": _pair(k), "v": _pair(v)} for q, k, v in per_layer] for per_layer in raw.calibration_qkv]
    test_qkv = [[(_pair(q), _pair(k), _pair(v)) for q, k, v in per_layer] for per_layer in raw.test_qkv]
    linear_cases, attention_cases = _choose_cases(raw, linear_count, attention_count)
    metadata = dict(raw.metadata)
    metadata.update({
        "linear_case_count": len(linear_cases),
        "attention_case_count": len(attention_cases),
        "nvfp4_mode": NVFP4_MODE,
        "input_codec": NVFP4_INPUT_CODEC,
        "case_design": "full-cartesian-real-wa-v3" if linear_count is None and attention_count is None else "explicit-smoke-prefix-v3",
        "linear_case_limit": linear_count,
        "attention_case_limit": attention_count,
        "linear_calibration_indices": list(linear_calibration_indices),
        "calibration_call_graph": "all-layer-role-once; attention-layer-once",
    })
    return PreparedPack(
        weights, cal_act, test_act, cal_qkv, test_qkv,
        raw.calibration_windows, linear_calibration_windows, raw.test_windows, raw.layers, raw.hidden_size,
        raw.q_heads, raw.kv_heads, raw.head_dim, linear_cases, attention_cases, metadata,
    )


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
    return {
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


def _attention_decomposition_summary(items: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {
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


def _group_summary(
    items: Sequence[Mapping[str, Any]],
    key: str,
    summary_fn: Any,
) -> dict[str, Any]:
    groups: dict[str, list[Mapping[str, Any]]] = {}
    for item in items:
        groups.setdefault(str(item[key]), []).append(item)
    return {name: summary_fn(group) for name, group in sorted(groups.items())}


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
    print(f"[{label}] Linear calibration: {pack.layers * len(ROLES)} layer/role states", flush=True)
    for layer in range(pack.layers):
        for role in ROLES:
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
        for role in ROLES
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
    return {
        "candidate": path.stem,
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
            "linear_role_macro_mean": sum(linear_by_role.values()) / len(ROLES),
            "attention_layer_macro_mean": sum(attention_by_layer.values()) / max(1, len(attention_by_layer)),
            "linear_cases": len(linear_details),
            "attention_cases": len(attention_details),
            "linear_by_role": linear_by_role,
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
    "v001": {"path": "solutions/20260826_v001_current-baseline_score10250_time127s/solution.py", "official_score": 10250, "official_time": 127.0, "official_status": "pass"},
    "v002": {"path": "solutions/20260826_v002_youxilee-hif4_score15000plus_timeNA/solution.py", "official_score": 15313, "official_time": 137.0, "official_status": "pass"},
    "v013": {"path": "solutions/20260827_v013_c10-wide-activation-quadratic_score15799_time144s/solution.py", "official_score": 15799, "official_time": 144.0, "official_status": "pass"},
    "v024": {"path": "solutions/20260827_v024_c21-gated-exact-cross-selection_score16043_time174s/solution.py", "official_score": 16043, "official_time": 173.8, "official_status": "pass"},
    "v025": {"path": "solutions/20260827_v025_c21c-compliance-baseline/solution.py", "official_score": 14437, "official_time": 166.6, "official_status": "pass"},
    "v030": {"path": "solutions/20260828_v030_c38-beam2-fullcov-official14092_time170.6s/solution.py", "official_score": 14092, "official_time": 170.57, "official_status": "pass"},
    "v031": {"path": "solutions/20260828_v031_c39-fw-official21864_time161.3s/solution.py", "official_score": 21864, "official_time": 161.3, "official_status": "pass"},
    "v032": {"path": "solutions/20260828_v032_c40-robust-blockldlq_official-score14432_time216.667s/solution.py", "official_score": 14432, "official_time": 216.667, "official_status": "pass"},
    "v034": {"path": "solutions/20260829_v034_c41b-mha-k-center_scoreNA_timeNA/solution.py", "official_score": 21864, "official_time": 159.4, "official_status": "pass"},
    "v051": {"path": "solutions/20260829_v051_c47b-grouping-threshold005_scoreNA_timeNA/solution.py", "official_score": 22451, "official_time": 234.0, "official_status": "pass"},
    "v066": {"path": "solutions/20260829_v066_c66-activation-ratio100_scoreNA_timeNA/solution.py", "official_score": 22557, "official_time": 217.2, "official_status": "pass"},
    "v072": {"path": "solutions/20260829_v072_c74-jdrq-hierarchy_scoreNA_timeNA/solution.py", "official_score": 22662, "official_time": 226.0, "official_status": "pass"},
    "v074": {"path": "solutions/20260829_v074_c75-rowwise-jdrq_scoreNA_timeNA/solution.py", "official_score": 22750, "official_time": 239.387, "official_status": "pass"},
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


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _write_report(path: Path, result: Mapping[str, Any], pack: PreparedPack) -> None:
    score = result["score"]
    timing = result["timing"]
    decomposition = result.get("decomposition", {})
    lines = [
        f"# {result['candidate']} — {PROTOCOL}",
        "",
        "本报告是本地 proxy；隐藏官方数据和鲲鹏硬件不可本地复制。",
        "",
        f"- calibration lengths: `{list(CALIBRATION_LENGTHS)}`",
        f"- cases: `{score['linear_cases']} Linear + {score['attention_cases']} Attention` (full captured W/A by default)",
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
        "",
        "## 误差源分解（evaluator-only）",
        "",
        "控制臂只在评测器内重算，不增加六个候选 API 调用，也不改变主分数。gain 为相对标准输出误差的改善；interaction 为正表示超加性互补，负表示收益重叠或递减。",
    ]
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
) -> dict[str, Any]:
    return {
        "protocol": PROTOCOL,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "data_source": data_source,
        "capture_seconds": capture_seconds,
        "cache": str(cache_path),
        "protocol_config": {
            "model": MODEL_NAME,
            "calibration_lengths": list(CALIBRATION_LENGTHS),
            "test_length": TEST_LENGTH,
            "test_lengths": list(TEST_LENGTHS),
            "test_windows": TEST_WINDOW_COUNT,
            "linear_cases": len(prepared.linear_cases),
            "attention_cases": len(prepared.attention_cases),
            "case_design_scope": "all captured W/A tensors by default; optional CLI limits are smoke-only",
            "case_design": prepared.metadata.get("case_design", "full-cartesian-real-wa-v3"),
            "calibration_call_graph": prepared.metadata.get(
                "calibration_call_graph", "all-layer-role-once; attention-layer-once"
            ),
            "input_codec": NVFP4_INPUT_CODEC,
            "runtime_limit_seconds": OFFICIAL_RUNTIME_LIMIT,
            "score_formula": "proxy gain=(MSE_STD-MSE_PLAYER)/MSE_STD per case; means are unweighted over actual captured cases",
            "runtime_measurement": "sum of elapsed six-API calls; wall_seconds is reported separately and neither is official time",
            "error_source_decomposition": "candidate API outputs are reused in evaluator-only Linear W/A and Attention Q/K/V control arms; it does not alter proxy means or API call counts",
            "trend_validation": "same-cohort pairwise ordering against user-confirmed anchors; diagnostic-only",
        },
        "data_metadata": prepared.metadata,
        "trend_diagnostics": _trend_diagnostics(results),
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
        f"- capture seconds: `{capture_seconds:.3f}`",
        f"- calibration lengths: `{list(CALIBRATION_LENGTHS)}`",
        f"- case counts: `{linear_count} Linear + {attention_count} Attention` (all captured W/A by default)",
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
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Any]:
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
        _write_json(args.output, _archive_output(data_source, capture_seconds, cache_path, prepared, results))
        if args.report:
            _write_archive_report(args.report, data_source, capture_seconds, results, prepared)
    output = _archive_output(data_source, capture_seconds, cache_path, prepared, results)
    _write_json(args.output, output)
    if args.report:
        _write_archive_report(args.report, data_source, capture_seconds, results, prepared)
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", action="store_true", help="evaluate every archived version with an official result")
    parser.add_argument("--solution", type=Path, help="one solution.py when --archive is not used")
    parser.add_argument("--name", default="candidate", help="name for --solution")
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--cache-mode", choices=("auto", "read", "write", "off"), default="auto")
    parser.add_argument(
        "--linear-cases",
        type=int,
        default=LINEAR_CASE_COUNT,
        help="optional Linear case limit for smoke only; default enumerates every captured W/A tensor",
    )
    parser.add_argument(
        "--attention-cases",
        type=int,
        default=ATTENTION_CASE_COUNT,
        help="optional Attention case limit for smoke only; default enumerates every captured W/A tensor",
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
    raise SystemExit(0 if output["results"] and all(item.get("status") == "ok" for item in output["results"]) else 1)
