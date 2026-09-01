"""Canonical local reproduction of the public HiF4 judging flow.

This is intentionally a new, single-protocol evaluator.  It does not import
the retired ``real_model_suite`` or any of the old ranking helpers.  The
protocol fixes the public judging geometry in one place:

* Qwen2.5-0.5B, all 24 transformer blocks;
* five variable-length calibration samples: 10, 128, 512, 1024, 1024 tokens;
* exactly 250 Linear and 200 Attention test cases;
* the public relative-MSE case score and end-to-end candidate timing.

The hidden official tensors and Kunpeng hardware are not available locally.
The generated public-data pack therefore uses pinned WikiText text and records
its source hash.  Shapes, call counts, state validation, HiF4 decoding, score
formula, and candidate API order are nevertheless the same as the published
judge contract.  No local seconds are converted to an official score or a
hardware pass/fail decision.
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


PROTOCOL = "official-shape-v1"
MODEL_NAME = "qwen2.5-0.5b"
MODEL_PATH = ROOT / "models" / MODEL_NAME
DATA_DIR = ROOT / "data" / "wikitext-2-raw-v1"
CACHE_DIR = ROOT / "artifacts" / "official_eval" / "cache"
DEFAULT_CACHE = CACHE_DIR / f"{MODEL_NAME}-{PROTOCOL}.pt"
CALIBRATION_LENGTHS = (10, 128, 512, 1024, 1024)
TEST_LENGTH = 128
TEST_WINDOW_COUNT = 9
LINEAR_CASE_COUNT = 250
ATTENTION_CASE_COUNT = 200
OFFICIAL_RUNTIME_LIMIT = 300.0
NVFP4_MODE = "amax6"
WIKITEXT_REVISION = "b08601e04326c79dfdd32d625aee71d232d685c3"
WIKITEXT_CONFIG = "wikitext-2-raw-v1"
WIKITEXT_FILES = {
    "train": "train-00000-of-00001.parquet",
    "validation": "validation-00000-of-00001.parquet",
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


@dataclasses.dataclass(frozen=True)
class Window:
    split: str
    document_id: str
    row_start: int
    row_end: int
    token_start: int
    token_end: int
    input_ids: tuple[int, ...]


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
    linear_cases: list[tuple[int, str, int]]
    attention_cases: list[tuple[int, int]]
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
        found: Window | None = None
        for document_id, row_start, row_end, ids in docs:
            if len(ids) < length:
                continue
            starts = (0, max(0, len(ids) - length), (sample_index * 97) % max(1, len(ids) - length + 1))
            for start in starts:
                end = start + length
                ranges = used_ranges.setdefault(document_id, [])
                if any(max(start, left) < min(end, right) for left, right in ranges):
                    continue
                found = Window("train", document_id, row_start, row_end, start, end, tuple(ids[start:end]))
                break
            if found is not None:
                break
        if found is None:
            raise RuntimeError(f"cannot select calibration window with length {length}")
        selected.append(found)
        used_ranges.setdefault(found.document_id, []).append((found.token_start, found.token_end))
    return selected


def _select_test_windows(tokenizer: Any, rows: Sequence[str]) -> list[Window]:
    docs = _tokenized_documents(tokenizer, rows, "validation")
    candidates: list[Window] = []
    for document_id, row_start, row_end, ids in docs:
        for start in range(0, len(ids) - TEST_LENGTH + 1, TEST_LENGTH):
            end = start + TEST_LENGTH
            candidates.append(Window("validation", document_id, row_start, row_end, start, end, tuple(ids[start:end])))
    if len(candidates) < TEST_WINDOW_COUNT:
        raise RuntimeError(f"WikiText validation has only {len(candidates)} test windows")
    # Round-robin source documents prevents one long article from dominating.
    by_document: dict[str, list[Window]] = {}
    for window in candidates:
        by_document.setdefault(window.document_id, []).append(window)
    selected: list[Window] = []
    round_index = 0
    while len(selected) < TEST_WINDOW_COUNT:
        added = False
        for values in by_document.values():
            if round_index < len(values):
                selected.append(values[round_index])
                added = True
                if len(selected) == TEST_WINDOW_COUNT:
                    break
        if not added:
            raise RuntimeError("could not select the requested test windows")
        round_index += 1
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
        calibration_windows = _select_calibration_windows(tokenizer, train_rows)
        test_windows = _select_test_windows(tokenizer, validation_rows)
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
            "test_window_count": len(test_windows),
            "capture_device": str(device),
            "weights_dtype": "float32",
            "weight_layout": "[out_features, in_features]",
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
        raise RuntimeError(f"data pack {path} is not protocol {PROTOCOL}")
    calibration_windows = [_window_from_payload(item) for item in payload["calibration_windows"]]
    test_windows = [_window_from_payload(item) for item in payload["test_windows"]]
    if [len(window.input_ids) for window in calibration_windows] != list(CALIBRATION_LENGTHS):
        raise RuntimeError("calibration data pack has wrong variable-length schedule")
    if any(len(window.input_ids) != TEST_LENGTH for window in test_windows):
        raise RuntimeError("test data pack has wrong test length")
    weights = payload["weights"]
    hidden_size = int(payload["hidden_size"])
    q_heads = int(payload["q_heads"])
    kv_heads = int(payload["kv_heads"])
    head_dim = int(payload["head_dim"])
    # Reject stale snapshots produced by the pre-v1 transpose bug.  The public
    # API receives native [out_features, in_features] weights.
    if not isinstance(weights, list) or not weights:
        raise RuntimeError("official data pack has no layer weights")
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
    return RawPack(
        weights, payload["calibration_activations"], payload["test_activations"],
        payload["calibration_qkv"], payload["test_qkv"], calibration_windows, test_windows,
        int(payload["layers"]), hidden_size, q_heads,
        kv_heads, head_dim, metadata,
    )


def _pair(value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    carrier, scale = nvfp4_encode(value, NVFP4_MODE)
    return carrier.contiguous(), scale.contiguous()


def _stable_key(value: Any) -> str:
    return hashlib.sha256(repr(value).encode("utf-8")).hexdigest()


def _choose_cases(pack: RawPack) -> tuple[list[tuple[int, str, int]], list[tuple[int, int]]]:
    linear_pool = [(layer, role, window) for layer in range(pack.layers) for role in ROLES for window in range(len(pack.test_windows))]
    attention_pool = [(layer, window) for layer in range(pack.layers) for window in range(len(pack.test_windows))]
    linear_pool.sort(key=_stable_key)
    attention_pool.sort(key=_stable_key)
    if len(linear_pool) < LINEAR_CASE_COUNT or len(attention_pool) < ATTENTION_CASE_COUNT:
        raise RuntimeError("official case pool is smaller than 250 Linear + 200 Attention")
    return linear_pool[:LINEAR_CASE_COUNT], attention_pool[:ATTENTION_CASE_COUNT]


def prepare_pack(raw: RawPack) -> PreparedPack:
    weights = [{role: _pair(value) for role, value in per_layer.items()} for per_layer in raw.weights]
    # The public Attention mini-sample is variable-length.  Linear's public
    # interface has a separate, small calibration list; keep two samples for
    # the Linear path instead of accidentally multiplying its work by the
    # Attention shape schedule.
    linear_calibration_indices = list(range(min(2, len(raw.calibration_windows))))
    linear_calibration_windows = [raw.calibration_windows[index] for index in linear_calibration_indices]
    cal_act = {role: [[_pair(raw.calibration_activations[role][sample][layer]) for layer in range(raw.layers)] for sample in linear_calibration_indices] for role in ROLES}
    test_act = {role: [[_pair(raw.test_activations[role][sample][layer]) for layer in range(raw.layers)] for sample in range(len(raw.test_windows))] for role in ROLES}
    cal_qkv = [[{"q": _pair(q), "k": _pair(k), "v": _pair(v)} for q, k, v in per_layer] for per_layer in raw.calibration_qkv]
    test_qkv = [[(_pair(q), _pair(k), _pair(v)) for q, k, v in per_layer] for per_layer in raw.test_qkv]
    linear_cases, attention_cases = _choose_cases(raw)
    metadata = dict(raw.metadata)
    metadata.update({
        "linear_case_count": len(linear_cases),
        "attention_case_count": len(attention_cases),
        "nvfp4_mode": NVFP4_MODE,
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


def _score(standard: torch.Tensor, player: torch.Tensor, reference: torch.Tensor) -> float:
    standard_mse = float((standard - reference).square().mean())
    player_mse = float((player - reference).square().mean())
    if not math.isfinite(standard_mse) or standard_mse <= 0.0:
        raise ValueError("MSE_STD must be finite and positive")
    if not math.isfinite(player_mse):
        raise ValueError("MSE_PLAYER must be finite")
    return (standard_mse - player_mse) / standard_mse


def _cpu_params(params: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {name: value.detach().to("cpu") for name, value in params.items()}


def evaluate_solution(path: Path, pack: PreparedPack, algorithm_device_name: str) -> dict[str, Any]:
    solution = load_solution(path)
    label = path.parent.name
    device = torch.device(algorithm_device_name)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    wall_start = time.perf_counter()
    api_seconds = {name: 0.0 for name in REQUIRED_APIS}
    api_calls = {name: 0 for name in REQUIRED_APIS}
    weight_states: dict[tuple[int, str], tuple[Any, dict[str, torch.Tensor]]] = {}
    attention_states: dict[int, dict[str, Any]] = {}
    linear_layers = sorted({layer for layer, _role, _window in pack.linear_cases})
    linear_roles = sorted({role for _layer, role, _window in pack.linear_cases})
    print(f"[{label}] Linear calibration: {len(linear_layers) * len(linear_roles)} weight groups", flush=True)
    for layer in linear_layers:
        for role in linear_roles:
            weight_pair = _move_pair(pack.weights[layer][role], device)
            calibration = [_move_pair(pack.linear_calibration_activations[role][sample][layer], device) for sample in range(len(pack.linear_calibration_windows))]
            started = time.perf_counter()
            result = solution.hif4_calibration_and_quantize_weight(weight_pair[0], weight_pair[1], calibration)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            elapsed = time.perf_counter() - started
            api_seconds["hif4_calibration_and_quantize_weight"] += elapsed
            api_calls["hif4_calibration_and_quantize_weight"] += 1
            if not isinstance(result, Mapping) or set(result) != {"weight_params", "activation_state"}:
                raise ValueError("weight calibration must return exactly weight_params and activation_state")
            validate_state(result["activation_state"])
            validate_hif4_params(
                result["weight_params"],
                dequantize_nvfp4(*pack.weights[layer][role]).shape,
            )
            weight_states[(layer, role)] = (result["activation_state"], _cpu_params(result["weight_params"]))
    print(f"[{label}] Attention calibration: {len({layer for layer, _window in pack.attention_cases})} layers", flush=True)
    for layer in sorted({layer for layer, _window in pack.attention_cases}):
        calibration = [_move_qkv(pack.calibration_qkv[sample][layer], device) for sample in range(len(pack.calibration_windows))]
        started = time.perf_counter()
        states = solution.hif4_calibration_attention(calibration, pack.q_heads, pack.kv_heads, pack.head_dim)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - started
        api_seconds["hif4_calibration_attention"] += elapsed
        api_calls["hif4_calibration_attention"] += 1
        if not isinstance(states, Mapping) or set(states) != {"q_state", "k_state", "v_state"}:
            raise ValueError("attention calibration must return exactly q_state, k_state, v_state")
        for name in ("q_state", "k_state", "v_state"):
            validate_state(states[name])
        attention_states[layer] = dict(states)

    linear_scores: list[float] = []
    attention_scores: list[float] = []
    standard_weight_cache: dict[tuple[int, str], torch.Tensor] = {}
    print(f"[{label}] Linear scoring: {len(pack.linear_cases)} cases", flush=True)
    for layer, role, window in pack.linear_cases:
        state, weight_params = weight_states[(layer, role)]
        activation_pair = _move_pair(pack.test_activations[role][window][layer], device)
        started = time.perf_counter()
        activation_params = solution.hif4_dynamic_quantize_activation(activation_pair[0], activation_pair[1], state)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        api_seconds["hif4_dynamic_quantize_activation"] += time.perf_counter() - started
        api_calls["hif4_dynamic_quantize_activation"] += 1
        ref_activation = dequantize_nvfp4(*pack.test_activations[role][window][layer]).to(torch.float32)
        ref_weight = dequantize_nvfp4(*pack.weights[layer][role]).to(torch.float32)
        if (layer, role) not in standard_weight_cache:
            standard_weight_cache[(layer, role)] = decode_standard_hif4(encode_standard_hif4(ref_weight)).to(torch.float32)
        standard_activation = decode_standard_hif4(encode_standard_hif4(ref_activation)).to(torch.float32)
        player_activation = dequantize_hif4(_cpu_params(activation_params), ref_activation.shape).to(torch.float32)
        player_weight = dequantize_hif4(weight_params, ref_weight.shape).to(torch.float32)
        reference = ref_activation @ ref_weight.T
        standard = standard_activation @ standard_weight_cache[(layer, role)].T
        player = player_activation @ player_weight.T
        linear_scores.append(_score(standard, player, reference))
    print(f"[{label}] Attention scoring: {len(pack.attention_cases)} cases", flush=True)
    for layer, window in pack.attention_cases:
        states = attention_states[layer]
        q_pair, k_pair, v_pair = (_move_pair(pair, device) for pair in pack.test_qkv[window][layer])
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
        ref_q = dequantize_nvfp4(*pack.test_qkv[window][layer][0]).to(torch.float32)
        ref_k = dequantize_nvfp4(*pack.test_qkv[window][layer][1]).to(torch.float32)
        ref_v = dequantize_nvfp4(*pack.test_qkv[window][layer][2]).to(torch.float32)
        std_q = decode_standard_hif4(encode_standard_hif4(ref_q)).to(torch.float32)
        std_k = decode_standard_hif4(encode_standard_hif4(ref_k)).to(torch.float32)
        std_v = decode_standard_hif4(encode_standard_hif4(ref_v)).to(torch.float32)
        player_q = dequantize_hif4(_cpu_params(q_params), ref_q.shape).to(torch.float32)
        player_k = dequantize_hif4(_cpu_params(k_params), ref_k.shape).to(torch.float32)
        player_v = dequantize_hif4(_cpu_params(v_params), ref_v.shape).to(torch.float32)
        reference = _attention(ref_q[None], ref_k[None], ref_v[None], pack.q_heads, pack.kv_heads, pack.head_dim)
        standard = _attention(std_q[None], std_k[None], std_v[None], pack.q_heads, pack.kv_heads, pack.head_dim)
        player = _attention(player_q[None], player_k[None], player_v[None], pack.q_heads, pack.kv_heads, pack.head_dim)
        attention_scores.append(_score(standard, player, reference))
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    wall_seconds = time.perf_counter() - wall_start
    linear_sum = float(sum(linear_scores))
    attention_sum = float(sum(attention_scores))
    return {
        "candidate": path.stem,
        "source": str(path.resolve()),
        "source_sha256": sha256_file(path),
        "score": {
            "linear_sum": linear_sum,
            "attention_sum": attention_sum,
            "total_sum": linear_sum + attention_sum,
            "linear_mean": linear_sum / LINEAR_CASE_COUNT,
            "attention_mean": attention_sum / ATTENTION_CASE_COUNT,
            "equal_weight_45000_scale": (linear_sum + attention_sum) * 100.0,
            "linear_cases": LINEAR_CASE_COUNT,
            "attention_cases": ATTENTION_CASE_COUNT,
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
    "v084": {"path": "solutions/20260830_v084_c84-gram64-sweep5_scoreNA_timeNA/solution.py", "official_score": 16517, "official_time": 252.563, "official_status": "pass"},
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
    lines = [
        f"# {result['candidate']} — {PROTOCOL}",
        "",
        "本报告使用唯一官方形状协议；隐藏官方数据和鲲鹏硬件不可本地复制。",
        "",
        f"- calibration lengths: `{list(CALIBRATION_LENGTHS)}`",
        f"- cases: `{LINEAR_CASE_COUNT} Linear + {ATTENTION_CASE_COUNT} Attention`",
        f"- source SHA256: `{result['source_sha256']}`",
        f"- data pack: `{pack.metadata.get('data_sha256', {})}`",
        "",
        "| 指标 | 值 |",
        "|---|---:|",
        f"| Linear mean | {score['linear_mean']:.9f} |",
        f"| Attention mean | {score['attention_mean']:.9f} |",
        f"| Equal-weight 45000 scale | {score['equal_weight_45000_scale']:.6f} |",
        f"| Candidate wall | {timing['wall_seconds']:.3f}s |",
        f"| Candidate API total | {timing['api_total_seconds']:.3f}s |",
        "",
        "官方成绩只保留为独立历史字段，不参与本地评分或时间换算。",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


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
            "test_windows": TEST_WINDOW_COUNT,
            "linear_cases": LINEAR_CASE_COUNT,
            "attention_cases": ATTENTION_CASE_COUNT,
            "runtime_limit_seconds": OFFICIAL_RUNTIME_LIMIT,
            "score_formula": "(MSE_STD-MSE_PLAYER)/MSE_STD per case; official total is the sum of case scores",
            "runtime_measurement": "sum of elapsed six-API calls; wall_seconds is reported separately and is not the official timer",
        },
        "data_metadata": prepared.metadata,
        "results": list(results),
    }


def _write_archive_report(path: Path, data_source: str, capture_seconds: float, results: Sequence[Mapping[str, Any]]) -> None:
    lines = [
        f"# {PROTOCOL} archive evaluation",
        "",
        f"- data source: `{data_source}`",
        f"- capture seconds: `{capture_seconds:.3f}`",
        f"- calibration lengths: `{list(CALIBRATION_LENGTHS)}`",
        f"- case counts: `{LINEAR_CASE_COUNT} Linear + {ATTENTION_CASE_COUNT} Attention`",
        "",
        "| Candidate | Status | Linear mean | Attention mean | Equal-weight scale | API total(s) | API<300 | Wall(s) | Wall<300 | Official status |",
        "|---|---|---:|---:|---:|---:|---|---:|---|---|",
    ]
    for result in results:
        score = result.get("score", {})
        timing = result.get("timing", {})
        if result.get("status") == "ok":
            linear_mean = f"{score['linear_mean']:.6f}"
            attention_mean = f"{score['attention_mean']:.6f}"
            equal_weight = f"{score['equal_weight_45000_scale']:.3f}"
            api_total = f"{timing['api_total_seconds']:.3f}"
            api_under = str(timing['local_api_under_300_indicator'])
            wall = f"{timing['wall_seconds']:.3f}"
            wall_under = str(timing['wall_under_300_indicator'])
        else:
            linear_mean = attention_mean = equal_weight = api_total = wall = "-"
            api_under = wall_under = "-"
        lines.append(
            f"| {result['candidate']} | {result['status']} | "
            f"{linear_mean} | {attention_mean} | {equal_weight} | {api_total} | "
            f"{api_under} | {wall} | {wall_under} | "
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
    prepared = prepare_pack(raw)
    capture_seconds = time.perf_counter() - capture_started
    if args.archive:
        candidates = [(name, item) for name, item in ARCHIVE_MANIFEST.items()]
    else:
        if args.solution is None:
            raise ValueError("--solution is required unless --archive is used")
        candidates = [(args.name, {"path": str(args.solution), "official_score": None, "official_time": None, "official_status": "unregistered"})]
    results: list[dict[str, Any]] = []
    for name, item in candidates:
        source = (ROOT / item["path"]).resolve()
        print(f"[{name}] evaluating {source}", flush=True)
        try:
            result = evaluate_solution(source, prepared, args.algorithm_device)
            result["candidate"] = name
            result["official"] = {
                "score": item.get("official_score"),
                "time_seconds": item.get("official_time"),
                "status": item.get("official_status"),
            }
            result["status"] = "ok"
        except Exception as exc:
            result = {
                "candidate": name,
                "source": str(source),
                "source_sha256": sha256_file(source) if source.is_file() else None,
                "official": {"score": item.get("official_score"), "time_seconds": item.get("official_time"), "status": item.get("official_status")},
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
            _write_archive_report(args.report, data_source, capture_seconds, results)
    output = _archive_output(data_source, capture_seconds, cache_path, prepared, results)
    _write_json(args.output, output)
    if args.report:
        _write_archive_report(args.report, data_source, capture_seconds, results)
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", action="store_true", help="evaluate every archived version with an official result")
    parser.add_argument("--solution", type=Path, help="one solution.py when --archive is not used")
    parser.add_argument("--name", default="candidate", help="name for --solution")
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--cache-mode", choices=("auto", "read", "write", "off"), default="auto")
    parser.add_argument("--capture-device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--algorithm-device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts" / "official_eval" / "archive.json")
    parser.add_argument("--report", type=Path, default=ROOT / "logs" / "official_eval" / "archive.md")
    return parser


if __name__ == "__main__":
    output = run(build_parser().parse_args())
    raise SystemExit(0 if output["results"] and all(item.get("status") == "ok" for item in output["results"]) else 1)
