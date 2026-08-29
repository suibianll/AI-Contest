"""Evaluate compliant HiF4 candidates on several real language models.

This is a development evaluator, not an official-score replacement.  It has
four deliberate properties:

* calibration windows come from WikiText-2 train and test windows come from
  WikiText-2 validation; windows never wrap, repeat, or share a source
  document;
* every activation is captured from an actual local ``AutoModelForCausalLM``
  forward pass, with model-specific projection and rotary-embedding adapters;
* the evaluator computes the reference outputs only after a candidate has
  returned its quantization state.  The candidate receives weights and
  activations, never an evaluator output, residual, or fitted official score.
* the default local ranking uses a Qwen-shaped, mean-preserving panel with
  250 Linear and 200 Attention slots.  Other model families remain soft
  guardrails instead of being summed by layer count.

Offline Linear calibration may form ``A @ W`` to optimize the offline weight
quantizer ``Q(W)``.  A solution must not route that output or residual into
``activation_state`` or use it to select or infer the online ``Q(A)``.  The
output products in this file are evaluator-side scoring references only.

Typical run (from the repository root)::

    python evaluator/real_model_suite.py --device cuda

Capture once, then score candidates offline from the snapshots::

    python evaluator/real_model_suite.py --device cuda --cache-mode write --capture-only
    python evaluator/real_model_suite.py --device cpu --algorithm-device cuda --cache-mode read

Model weights and WikiText parquet files are local, ignored assets.  The
manifest pins the public Hugging Face revisions used to obtain them; the
download itself is intentionally performed outside this script so an
offline evaluator cannot silently change the corpus.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
import re
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Sequence

import torch

ROOT = Path(__file__).resolve().parents[1]
EVALUATOR_DIR = Path(__file__).resolve().parent
if str(EVALUATOR_DIR) not in sys.path:
    sys.path.insert(0, str(EVALUATOR_DIR))

from nvfp4_sim import nvfp4_encode  # noqa: E402
from real_data_eval import (  # noqa: E402
    causal_attention,
    instrument_solution,
    load_solution,
    std_hif4,
)
from reference_hif4 import (  # noqa: E402
    dequantize_hif4,
    dequantize_nvfp4,
    validate_state,
)


WIKITEXT_REVISION = "b08601e04326c79dfdd32d625aee71d232d685c3"
WIKITEXT_CONFIG = "wikitext-2-raw-v1"
WIKITEXT_FILES = {
    "train": "train-00000-of-00001.parquet",
    "validation": "validation-00000-of-00001.parquet",
    "test": "test-00000-of-00001.parquet",
}
CACHE_SCHEMA_VERSION = 1
SCORING_PROTOCOL_VERSION = 3
OFFICIAL_RUNTIME_LIMIT_SECONDS = 420.0
OFFICIAL_PANEL_REVISION = "2026-08-29"
REFERENCE_PANEL_LINEAR_CASES = 250
REFERENCE_PANEL_ATTENTION_CASES = 200
REFERENCE_PANEL_TOTAL_CASES = (
    REFERENCE_PANEL_LINEAR_CASES + REFERENCE_PANEL_ATTENTION_CASES
)
DEFAULT_PANEL_PROFILE = "qwen-official"
PANEL_PROFILES = ("qwen-official", "native")
DEFAULT_PRIMARY_MODEL = "qwen2.5-0.5b"
EXTERNAL_OFFICIAL_REFERENCES = (
    {
        "name": "youxilee/hif4",
        "score": 24153,
        "time_seconds": 239.0,
        "url": "https://github.com/youxilee/hif4",
        "imported_as_candidate": False,
    },
)
OFFICIAL_EXTRA_CASE_REFERENCE = {
    "count": 2,
    "architecture_hint": "Qwen 30B-like",
    "local_inputs_available": False,
    "use": "diagnostic shape guidance only; never a score-fitting target",
}
DEFAULT_CACHE_DIR = ROOT / "artifacts" / "real_model_suite" / "cache"
STANDARD_CODEC_PATH = EVALUATOR_DIR / "reference_hif4.py"


@dataclasses.dataclass(frozen=True)
class ModelSpec:
    name: str
    family: str
    path: Path
    source_revision: str


MODEL_SPECS: dict[str, ModelSpec] = {
    "gpt2-small": ModelSpec(
        "gpt2-small", "gpt2", ROOT / "models" / "gpt2", "openai-community/gpt2@main"
    ),
    "gpt2-medium": ModelSpec(
        "gpt2-medium",
        "gpt2",
        ROOT / "models" / "gpt2-medium",
        "openai-community/gpt2-medium@main",
    ),
    "opt-125m": ModelSpec(
        "opt-125m", "opt", ROOT / "models" / "opt-125m", "facebook/opt-125m@27dcfa74d334bc871f3234de431e71c6eeba5dd6"
    ),
    "pythia-160m": ModelSpec(
        "pythia-160m",
        "gpt_neox",
        ROOT / "models" / "pythia-160m",
        "EleutherAI/pythia-160m@50f5173d932e8e61f858120bcb800b97af589f46",
    ),
    "qwen2.5-0.5b": ModelSpec(
        "qwen2.5-0.5b",
        "qwen2",
        ROOT / "models" / "qwen2.5-0.5b",
        "Qwen/Qwen2.5-0.5B@060db6499f32faf8b98477b0a26969ef7d8b9987",
    ),
}


@dataclasses.dataclass(frozen=True)
class CandidateSpec:
    name: str
    path: Path
    official_score: int | None
    official_time: float | None
    # Historical scores are intentionally retained for provenance, but only
    # anchors carrying the current panel revision participate in audit metrics.
    official_panel_revision: str | None = None


CANDIDATE_SPECS: dict[str, CandidateSpec] = {
    "v001": CandidateSpec(
        "v001",
        ROOT / "solutions" / "20260826_v001_current-baseline_score10250_time127s" / "solution.py",
        10250,
        127.0,
    ),
    "v002": CandidateSpec(
        "v002",
        ROOT / "solutions" / "20260826_v002_youxilee-hif4_score15000plus_timeNA" / "solution.py",
        15313,
        137.0,
    ),
    "v013": CandidateSpec(
        "v013",
        ROOT / "solutions" / "20260827_v013_c10-wide-activation-quadratic_score15799_time144s" / "solution.py",
        15799,
        144.0,
    ),
    # v024 (C21) earned 16043 via a Linear output-supervision path that the
    # compliance review later rejected; it is kept only as a historical anchor.
    "v024": CandidateSpec(
        "v024",
        ROOT / "solutions" / "20260827_v024_c21-gated-exact-cross-selection_score16043_time174s" / "solution.py",
        16043,
        173.8,
    ),
    "c21": CandidateSpec(
        "c21",
        ROOT / "solutions" / "20260827_v025_c21c-compliance-baseline" / "solution.py",
        14437,
        166.6,
    ),
    "c38": CandidateSpec(
        "c38",
        ROOT / "solutions" / "20260828_v030_c38-beam2-fullcov-official14092_time170.6s" / "solution.py",
        14092,
        170.57,
    ),
    "c39": CandidateSpec(
        "c39",
        ROOT / "solutions" / "20260828_v031_c39-fw-official14613_time159.2s" / "solution.py",
        21864,
        161.3,
        OFFICIAL_PANEL_REVISION,
    ),
    "c40": CandidateSpec(
        "c40",
        ROOT / "solutions" / "20260828_v032_c40-robust-blockldlq_official-score14432_time216.667s" / "solution.py",
        14432,
        216.667,
    ),
    "c41b": CandidateSpec(
        "c41b",
        ROOT / "solutions" / "20260829_v034_c41b-mha-k-center_scoreNA_timeNA" / "solution.py",
        21864,
        159.4,
        OFFICIAL_PANEL_REVISION,
    ),
    "c47b": CandidateSpec(
        "c47b",
        ROOT / "solutions" / "20260829_v051_c47b-grouping-threshold005_scoreNA_timeNA" / "solution.py",
        22451,
        234.0,
        OFFICIAL_PANEL_REVISION,
    ),
    "c66": CandidateSpec(
        "c66",
        ROOT / "solutions" / "20260829_v066_c66-activation-ratio100_scoreNA_timeNA" / "solution.py",
        22557,
        217.2,
        OFFICIAL_PANEL_REVISION,
    ),
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


@dataclasses.dataclass
class ModelData:
    spec: ModelSpec
    tokenizer_name: str
    layers: int
    hidden_size: int
    q_heads: int
    kv_heads: int
    head_dim: int
    roles: tuple[str, ...]
    role_groups: dict[str, str]
    weights: list[dict[str, torch.Tensor]]
    calibration_activations: dict[str, list[list[torch.Tensor]]]
    test_activations: dict[str, list[list[torch.Tensor]]]
    calibration_qkv: list[list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]]
    test_qkv: list[list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]]
    calibration_windows: list[Window]
    test_windows: list[Window]
    metadata: dict[str, Any]


class CacheValidationError(RuntimeError):
    """Raised when a persisted model snapshot is absent, stale, or malformed."""


def model_cache_path(
    spec: ModelSpec,
    cache_dir: Path,
    sequence_length: int,
    calibration_samples: int,
    test_samples: int,
    requested_layers: int | None,
) -> Path:
    """Return the deterministic path for one model/data capture configuration."""

    layer_tag = "all" if requested_layers is None else str(requested_layers)
    filename = (
        f"{spec.name}__seq{sequence_length}__calib{calibration_samples}"
        f"__test{test_samples}__layers{layer_tag}__schema{CACHE_SCHEMA_VERSION}.pt"
    )
    return cache_dir / filename


def _cache_capture_config(data: ModelData, requested_layers: int | None) -> dict[str, Any]:
    data_metadata = data.metadata.get("data", {})
    return {
        "model": data.spec.name,
        "family": data.spec.family,
        "source_revision": data.spec.source_revision,
        "dataset": data_metadata.get("dataset"),
        "dataset_config": data_metadata.get("config"),
        "dataset_revision": data_metadata.get("revision"),
        "sequence_length": len(data.calibration_windows[0].input_ids),
        "calibration_samples": len(data.calibration_windows),
        "test_samples": len(data.test_windows),
        "requested_layers": requested_layers,
        "layers": data.layers,
    }


def _window_to_cache_payload(window: Window) -> dict[str, Any]:
    payload = dataclasses.asdict(window)
    payload["input_ids"] = list(window.input_ids)
    return payload


def _window_from_cache_payload(payload: Any, label: str) -> Window:
    if not isinstance(payload, dict):
        raise CacheValidationError(f"{label} is not a serialized window mapping")
    try:
        input_ids = tuple(int(token) for token in payload["input_ids"])
        return Window(
            split=str(payload["split"]),
            document_id=str(payload["document_id"]),
            row_start=int(payload["row_start"]),
            row_end=int(payload["row_end"]),
            token_start=int(payload["token_start"]),
            token_end=int(payload["token_end"]),
            input_ids=input_ids,
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise CacheValidationError(f"malformed {label}") from exc


def _validate_cached_tensor(value: Any, label: str) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise CacheValidationError(f"{label} is not a tensor")
    if value.device.type != "cpu":
        raise CacheValidationError(f"{label} is not stored on CPU")
    if value.numel() == 0:
        raise CacheValidationError(f"{label} is empty")
    if value.is_floating_point() and not bool(torch.isfinite(value).all()):
        raise CacheValidationError(f"{label} contains NaN or infinity")
    return value


def _validate_activation_store(
    store: Any,
    roles: Sequence[str],
    sample_count: int,
    layer_count: int,
    hidden_size: int,
    label: str,
) -> None:
    if not isinstance(store, dict):
        raise CacheValidationError(f"{label} is not a role mapping")
    if set(store) != set(roles):
        raise CacheValidationError(f"{label} roles do not match the cached model")
    for role in roles:
        batches = store[role]
        if not isinstance(batches, list) or len(batches) != sample_count:
            raise CacheValidationError(
                f"{label}[{role}] has {len(batches) if isinstance(batches, list) else 'invalid'} samples"
            )
        for batch_index, per_layer in enumerate(batches):
            if not isinstance(per_layer, list) or len(per_layer) != layer_count:
                raise CacheValidationError(
                    f"{label}[{role}][{batch_index}] has the wrong layer count"
                )
            for layer_index, value in enumerate(per_layer):
                tensor = _validate_cached_tensor(
                    value, f"{label}[{role}][{batch_index}][{layer_index}]"
                )
                if tensor.ndim != 2 or tensor.shape[0] <= 0 or tensor.shape[1] <= 0:
                    raise CacheValidationError(
                        f"{label}[{role}][{batch_index}][{layer_index}] has invalid shape {tuple(tensor.shape)}"
                    )
                if role in {"q", "k", "v", "o"} and tensor.shape[1] != hidden_size:
                    raise CacheValidationError(
                        f"{label}[{role}][{batch_index}][{layer_index}] has input width "
                        f"{tensor.shape[1]}, expected {hidden_size}"
                    )


def _validate_qkv_store(
    store: Any,
    sample_count: int,
    layer_count: int,
    q_width: int,
    kv_width: int,
    label: str,
) -> None:
    if not isinstance(store, list) or len(store) != sample_count:
        raise CacheValidationError(f"{label} has the wrong sample count")
    for batch_index, per_layer in enumerate(store):
        if not isinstance(per_layer, list) or len(per_layer) != layer_count:
            raise CacheValidationError(f"{label}[{batch_index}] has the wrong layer count")
        for layer_index, qkv in enumerate(per_layer):
            if not isinstance(qkv, (tuple, list)) or len(qkv) != 3:
                raise CacheValidationError(f"{label}[{batch_index}][{layer_index}] is not Q/K/V")
            for name, value, width in zip(("q", "k", "v"), qkv, (q_width, kv_width, kv_width)):
                tensor = _validate_cached_tensor(
                    value, f"{label}[{batch_index}][{layer_index}].{name}"
                )
                if tensor.ndim != 2 or tensor.shape[1] != width:
                    raise CacheValidationError(
                        f"{label}[{batch_index}][{layer_index}].{name} has shape {tuple(tensor.shape)}, "
                        f"expected [tokens, {width}]"
                    )


def _validate_cached_model_payload(
    payload: Any,
    spec: ModelSpec,
    sequence_length: int,
    calibration_samples: int,
    test_samples: int,
    requested_layers: int | None,
) -> ModelData:
    if not isinstance(payload, dict):
        raise CacheValidationError("cache root is not a mapping")
    if payload.get("schema_version") != CACHE_SCHEMA_VERSION:
        raise CacheValidationError(
            f"unsupported cache schema {payload.get('schema_version')!r}; "
            f"expected {CACHE_SCHEMA_VERSION}"
        )

    capture_config = payload.get("capture_config")
    expected_config = {
        "model": spec.name,
        "family": spec.family,
        "source_revision": spec.source_revision,
        "dataset": "Salesforce/wikitext",
        "dataset_config": WIKITEXT_CONFIG,
        "dataset_revision": WIKITEXT_REVISION,
        "sequence_length": sequence_length,
        "calibration_samples": calibration_samples,
        "test_samples": test_samples,
        "requested_layers": requested_layers,
    }
    if not isinstance(capture_config, dict):
        raise CacheValidationError("cache has no capture_config")
    mismatches = {
        key: (capture_config.get(key), expected)
        for key, expected in expected_config.items()
        if capture_config.get(key) != expected
    }
    if mismatches:
        raise CacheValidationError(f"cache configuration mismatch: {mismatches}")

    metadata = payload.get("metadata")
    if not isinstance(metadata, dict):
        raise CacheValidationError("cache has no model metadata")
    data_metadata = metadata.get("data")
    if not isinstance(data_metadata, dict):
        raise CacheValidationError("cache has no dataset metadata")
    for key, expected in (
        ("model", spec.name),
        ("family", spec.family),
        ("source_revision", spec.source_revision),
    ):
        if metadata.get(key) != expected:
            raise CacheValidationError(
                f"cache metadata {key}={metadata.get(key)!r} does not match {expected!r}"
            )
    for key, expected in (
        ("dataset", "Salesforce/wikitext"),
        ("config", WIKITEXT_CONFIG),
        ("revision", WIKITEXT_REVISION),
    ):
        if data_metadata.get(key) != expected:
            raise CacheValidationError(
                f"cache data metadata {key}={data_metadata.get(key)!r} does not match {expected!r}"
            )

    try:
        layers = int(payload["layers"])
        hidden_size = int(payload["hidden_size"])
        q_heads = int(payload["q_heads"])
        kv_heads = int(payload["kv_heads"])
        head_dim = int(payload["head_dim"])
        roles = tuple(str(role) for role in payload["roles"])
        role_groups = {str(key): str(value) for key, value in payload["role_groups"].items()}
        tokenizer_name = str(payload["tokenizer_name"])
    except (KeyError, TypeError, ValueError, AttributeError) as exc:
        raise CacheValidationError("cache model description is malformed") from exc
    if layers <= 0 or hidden_size <= 0 or q_heads <= 0 or kv_heads <= 0 or head_dim <= 0:
        raise CacheValidationError("cache model dimensions must be positive")
    if q_heads % kv_heads or q_heads * head_dim != hidden_size:
        raise CacheValidationError("cache attention dimensions are inconsistent")
    if not roles or set(role_groups) != set(roles):
        raise CacheValidationError("cache roles and role_groups are inconsistent")

    calibration_windows_payload = payload.get("calibration_windows")
    test_windows_payload = payload.get("test_windows")
    if not isinstance(calibration_windows_payload, list) or not isinstance(test_windows_payload, list):
        raise CacheValidationError("cache windows are missing")
    calibration_windows = [
        _window_from_cache_payload(item, f"calibration_windows[{index}]")
        for index, item in enumerate(calibration_windows_payload)
    ]
    test_windows = [
        _window_from_cache_payload(item, f"test_windows[{index}]")
        for index, item in enumerate(test_windows_payload)
    ]
    if len(calibration_windows) != calibration_samples or len(test_windows) != test_samples:
        raise CacheValidationError("cache windows do not match the requested sample counts")
    if any(window.split != "train" for window in calibration_windows):
        raise CacheValidationError("calibration cache windows must come from train")
    if any(window.split != "validation" for window in test_windows):
        raise CacheValidationError("test cache windows must come from validation")
    try:
        validate_window_split(calibration_windows, test_windows, sequence_length)
    except ValueError as exc:
        raise CacheValidationError(f"cache window validation failed: {exc}") from exc

    weights = payload.get("weights")
    if not isinstance(weights, list) or len(weights) != layers:
        raise CacheValidationError("cache weights have the wrong layer count")
    for layer_index, per_role in enumerate(weights):
        if not isinstance(per_role, dict) or set(per_role) != set(roles):
            raise CacheValidationError(f"weights[{layer_index}] roles do not match")
        for role in roles:
            tensor = _validate_cached_tensor(per_role[role], f"weights[{layer_index}][{role}]")
            if tensor.ndim != 2 or tensor.shape[0] <= 0 or tensor.shape[1] <= 0:
                raise CacheValidationError(
                    f"weights[{layer_index}][{role}] has shape {tuple(tensor.shape)}"
                )

    calibration_activations = payload.get("calibration_activations")
    test_activations = payload.get("test_activations")
    _validate_activation_store(
        calibration_activations, roles, calibration_samples, layers, hidden_size, "calibration_activations"
    )
    _validate_activation_store(
        test_activations, roles, test_samples, layers, hidden_size, "test_activations"
    )
    for layer_index, per_role in enumerate(weights):
        for role in roles:
            expected_input_width = calibration_activations[role][0][layer_index].shape[1]
            if per_role[role].shape[1] != expected_input_width:
                raise CacheValidationError(
                    f"weights[{layer_index}][{role}] input width {per_role[role].shape[1]} "
                    f"does not match activation width {expected_input_width}"
                )
    q_width = q_heads * head_dim
    kv_width = kv_heads * head_dim
    calibration_qkv = payload.get("calibration_qkv")
    test_qkv = payload.get("test_qkv")
    _validate_qkv_store(
        calibration_qkv, calibration_samples, layers, q_width, kv_width, "calibration_qkv"
    )
    _validate_qkv_store(test_qkv, test_samples, layers, q_width, kv_width, "test_qkv")

    restored_metadata = dict(metadata)
    return ModelData(
        spec=spec,
        tokenizer_name=tokenizer_name,
        layers=layers,
        hidden_size=hidden_size,
        q_heads=q_heads,
        kv_heads=kv_heads,
        head_dim=head_dim,
        roles=roles,
        role_groups=role_groups,
        weights=weights,
        calibration_activations=calibration_activations,
        test_activations=test_activations,
        calibration_qkv=calibration_qkv,
        test_qkv=test_qkv,
        calibration_windows=calibration_windows,
        test_windows=test_windows,
        metadata=restored_metadata,
    )


def save_model_cache(data: ModelData, path: Path, requested_layers: int | None) -> None:
    """Persist a CPU-only model snapshot atomically for later offline scoring."""

    payload = {
        "schema_version": CACHE_SCHEMA_VERSION,
        "capture_config": _cache_capture_config(data, requested_layers),
        "tokenizer_name": data.tokenizer_name,
        "layers": data.layers,
        "hidden_size": data.hidden_size,
        "q_heads": data.q_heads,
        "kv_heads": data.kv_heads,
        "head_dim": data.head_dim,
        "roles": list(data.roles),
        "role_groups": dict(data.role_groups),
        "weights": data.weights,
        "calibration_activations": data.calibration_activations,
        "test_activations": data.test_activations,
        "calibration_qkv": data.calibration_qkv,
        "test_qkv": data.test_qkv,
        "calibration_windows": [
            _window_to_cache_payload(window) for window in data.calibration_windows
        ],
        "test_windows": [_window_to_cache_payload(window) for window in data.test_windows],
        "metadata": data.metadata,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(path.name + ".tmp")
    try:
        torch.save(payload, temporary_path)
        temporary_path.replace(path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def load_model_cache(
    path: Path,
    spec: ModelSpec,
    sequence_length: int,
    calibration_samples: int,
    test_samples: int,
    requested_layers: int | None,
) -> ModelData:
    """Load and validate a snapshot without importing transformers or loading a model."""

    if not path.is_file():
        raise FileNotFoundError(f"model cache does not exist: {path}")
    try:
        try:
            payload = torch.load(path, map_location="cpu", weights_only=True)
        except TypeError:  # compatibility with older torch releases
            payload = torch.load(path, map_location="cpu")
    except Exception as exc:
        raise CacheValidationError(f"cannot read model cache {path}: {exc}") from exc
    data = _validate_cached_model_payload(
        payload,
        spec,
        sequence_length,
        calibration_samples,
        test_samples,
        requested_layers,
    )
    data.metadata = dict(data.metadata)
    data.metadata["cache_path"] = str(path)
    data.metadata["cache_schema_version"] = CACHE_SCHEMA_VERSION
    data.metadata["loaded_from_cache"] = True
    return data


_TITLE_RE = re.compile(r"^\s*=+\s+.*?\s+=+\s*$")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_text_rows(path: Path) -> list[str]:
    """Load the fixed parquet text column without the datasets cache."""

    try:
        import pyarrow.parquet as parquet
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "pyarrow is required for the pinned WikiText parquet files; "
            "install evaluator/requirements.txt"
        ) from exc
    table = parquet.read_table(path, columns=["text"])
    return [str(value) if value is not None else "" for value in table["text"].to_pylist()]


def _split_documents(rows: Sequence[str], split: str) -> list[tuple[str, int, int, str]]:
    """Group WikiText raw rows into deterministic article documents."""

    documents: list[tuple[str, int, int, str]] = []
    start_row: int | None = None
    title = "untitled"
    body: list[str] = []

    def flush(end_row: int) -> None:
        nonlocal start_row, title, body
        text = "\n".join(body).strip()
        if start_row is not None and text:
            document_id = f"{split}:{start_row}:{title}"
            documents.append((document_id, start_row, end_row, text))
        start_row = None
        title = "untitled"
        body = []

    for row_index, raw in enumerate(rows):
        text = raw.strip()
        if _TITLE_RE.match(text):
            flush(row_index - 1)
            start_row = row_index
            title = text
            body = [text]
        elif text:
            if start_row is None:
                start_row = row_index
            body.append(text)
    flush(len(rows) - 1)
    if not documents:
        raise ValueError(f"WikiText {split} split contains no non-empty documents")
    return documents


def _tokenize_documents(
    tokenizer: Any,
    rows: Sequence[str],
    split: str,
    sequence_length: int,
) -> list[tuple[str, int, int, list[int]]]:
    documents = _split_documents(rows, split)
    tokenized: list[tuple[str, int, int, list[int]]] = []
    for document_id, row_start, row_end, text in documents:
        # We intentionally tokenize the complete article before making fixed
        # windows.  Temporarily lifting the tokenizer warning threshold avoids
        # a misleading "sequence too long" message; the model only receives
        # the resulting 128-token windows below.
        previous_max_length = getattr(tokenizer, "model_max_length", None)
        if previous_max_length is not None:
            tokenizer.model_max_length = 10**9
        try:
            encoded = tokenizer(
                text, add_special_tokens=False, return_attention_mask=False
            )
        finally:
            if previous_max_length is not None:
                tokenizer.model_max_length = previous_max_length
        token_ids = encoded["input_ids"]
        if token_ids and isinstance(token_ids[0], list):
            token_ids = token_ids[0]
        token_ids = [int(token) for token in token_ids]
        if len(token_ids) >= sequence_length:
            tokenized.append((document_id, row_start, row_end, token_ids))
    if not tokenized:
        raise ValueError(
            f"WikiText {split} has no document with {sequence_length} tokens"
        )
    return tokenized


def _select_windows(
    tokenizer: Any,
    rows: Sequence[str],
    split: str,
    sequence_length: int,
    count: int,
) -> list[Window]:
    """Select non-overlapping windows by round-robin over source documents."""

    if count <= 0:
        raise ValueError("window count must be positive")
    documents = _tokenize_documents(tokenizer, rows, split, sequence_length)
    candidates: list[list[Window]] = []
    for document_id, row_start, row_end, token_ids in documents:
        windows = []
        for token_start in range(0, len(token_ids) - sequence_length + 1, sequence_length):
            token_end = token_start + sequence_length
            windows.append(
                Window(
                    split,
                    document_id,
                    row_start,
                    row_end,
                    token_start,
                    token_end,
                    tuple(token_ids[token_start:token_end]),
                )
            )
        if windows:
            candidates.append(windows)

    selected: list[Window] = []
    round_index = 0
    while len(selected) < count:
        added = False
        for document_windows in candidates:
            if round_index < len(document_windows):
                selected.append(document_windows[round_index])
                added = True
                if len(selected) == count:
                    break
        if not added:
            raise ValueError(
                f"WikiText {split} has only {len(selected)} disjoint windows; "
                f"{count} are required"
            )
        round_index += 1
    return selected


def validate_window_split(
    calibration: Sequence[Window], test: Sequence[Window], sequence_length: int
) -> None:
    """Reject leakage and wrapping before a model forward pass is allowed."""

    if not calibration or not test:
        raise ValueError("calibration and test windows must both be non-empty")
    if any(len(window.input_ids) != sequence_length for window in (*calibration, *test)):
        raise ValueError("a selected window has the wrong sequence length")
    calibration_sources = {window.document_id for window in calibration}
    test_sources = {window.document_id for window in test}
    overlap = calibration_sources & test_sources
    if overlap:
        raise ValueError(f"calibration/test source-document leakage: {sorted(overlap)}")

    seen: dict[str, list[tuple[int, int]]] = {}
    for window in (*calibration, *test):
        ranges = seen.setdefault(window.document_id, [])
        current = (window.token_start, window.token_end)
        if any(max(current[0], other[0]) < min(current[1], other[1]) for other in ranges):
            raise ValueError(f"overlapping token windows in {window.document_id}")
        ranges.append(current)


def load_real_windows(
    tokenizer: Any,
    data_dir: Path,
    sequence_length: int,
    calibration_samples: int,
    test_samples: int,
) -> tuple[list[Window], list[Window], dict[str, Any]]:
    paths = {
        split: data_dir / filename for split, filename in WIKITEXT_FILES.items()
    }
    required = ("train", "validation")
    missing = [str(paths[split]) for split in required if not paths[split].is_file()]
    if missing:
        raise FileNotFoundError(
            "missing pinned WikiText parquet files: " + ", ".join(missing)
        )
    rows = {split: _load_text_rows(paths[split]) for split in required}
    calibration = _select_windows(
        tokenizer, rows["train"], "train", sequence_length, calibration_samples
    )
    test = _select_windows(
        tokenizer, rows["validation"], "validation", sequence_length, test_samples
    )
    validate_window_split(calibration, test, sequence_length)
    metadata = {
        "dataset": "Salesforce/wikitext",
        "config": WIKITEXT_CONFIG,
        "revision": WIKITEXT_REVISION,
        "files": {
            split: {
                "path": str(path),
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
                "rows": len(rows[split]),
            }
            for split, path in paths.items()
            if split in rows and path.is_file()
        },
        "calibration_documents": sorted({window.document_id for window in calibration}),
        "test_documents": sorted({window.document_id for window in test}),
        "calibration_windows": [dataclasses.asdict(window) | {"input_ids": None} for window in calibration],
        "test_windows": [dataclasses.asdict(window) | {"input_ids": None} for window in test],
        "no_source_document_overlap": True,
        "no_token_window_overlap": True,
    }
    return calibration, test, metadata


def _cpu_float(value: torch.Tensor) -> torch.Tensor:
    return value.detach().to(device="cpu", dtype=torch.float32).contiguous()


def _module_weight(module: torch.nn.Module, transpose: bool = False) -> torch.Tensor:
    weight = _cpu_float(module.weight)
    return weight.t().contiguous() if transpose else weight


def _module_input(captured: dict[str, torch.Tensor], key: str) -> torch.Tensor:
    actual_key = key if key in captured else f"{key}_in"
    if actual_key not in captured:
        raise RuntimeError(f"forward hook did not capture {key}")
    value = captured[actual_key]
    return value.reshape(-1, value.shape[-1]).contiguous()


def _module_output(captured: dict[str, torch.Tensor], key: str) -> torch.Tensor:
    actual_key = key if key in captured else f"{key}_out"
    if actual_key not in captured:
        raise RuntimeError(f"forward hook did not capture {key}")
    value = captured[actual_key]
    return value.reshape(-1, value.shape[-1]).contiguous()


def _capture_input_output(
    module: torch.nn.Module,
    context: dict[str, Any],
    key: str,
    handles: list[Any],
    layer_index: int,
    capture_output: bool = False,
) -> None:
    def hook(_module: torch.nn.Module, inputs: tuple[Any, ...], output: Any) -> None:
        layer_capture = context["current_layers"][layer_index]
        if not inputs:
            raise RuntimeError(f"module {key} received no inputs")
        layer_capture[f"{key}_in"] = _cpu_float(inputs[0])
        if capture_output:
            if isinstance(output, (tuple, list)):
                output = output[0]
            layer_capture[f"{key}_out"] = _cpu_float(output)

    handles.append(module.register_forward_hook(hook))


class ModelAdapter:
    """Architecture-specific projection and attention extraction contract."""

    roles: tuple[str, ...] = ("q", "k", "v", "o", "fc", "proj")
    role_groups: dict[str, str] = {
        "q": "q",
        "k": "k",
        "v": "v",
        "o": "o",
        "fc": "fc",
        "proj": "proj",
    }

    def __init__(self, model: torch.nn.Module, requested_layers: int | None) -> None:
        self.model = model
        self.blocks = self._find_blocks(model)
        if requested_layers is not None:
            self.blocks = self.blocks[:requested_layers]
        if not self.blocks:
            raise ValueError("model adapter selected zero layers")
        self.hidden_size = self._hidden_size(model)
        self.q_heads = self._q_heads(model)
        self.kv_heads = self._kv_heads(model)
        self.head_dim = self._head_dim(model)
        if self.q_heads % self.kv_heads != 0:
            raise ValueError("q_heads must be divisible by kv_heads")
        if self.q_heads * self.head_dim != self.hidden_size:
            raise ValueError("q_heads * head_dim must equal model hidden size")

    @staticmethod
    def _find_blocks(model: torch.nn.Module) -> list[torch.nn.Module]:
        raise NotImplementedError

    @staticmethod
    def _hidden_size(model: torch.nn.Module) -> int:
        raise NotImplementedError

    @staticmethod
    def _q_heads(model: torch.nn.Module) -> int:
        raise NotImplementedError

    @staticmethod
    def _kv_heads(model: torch.nn.Module) -> int:
        return ModelAdapter._q_heads(model)

    @staticmethod
    def _head_dim(model: torch.nn.Module) -> int:
        config = model.config
        if getattr(config, "head_dim", None) is not None:
            return int(config.head_dim)
        hidden = getattr(config, "hidden_size", getattr(config, "n_embd", None))
        return int(hidden) // int(config.num_attention_heads)

    def register_hooks(self, context: dict[str, Any], handles: list[Any]) -> None:
        raise NotImplementedError

    def weights(self, block: torch.nn.Module) -> dict[str, torch.Tensor]:
        raise NotImplementedError

    def finalize(
        self, captured: dict[str, torch.Tensor], rope: Any
    ) -> tuple[dict[str, torch.Tensor], tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        raise NotImplementedError

    def register_rope_hook(self, context: dict[str, Any], handles: list[Any]) -> None:
        del context, handles

    @property
    def attention_available(self) -> bool:
        return True


class GPT2Adapter(ModelAdapter):
    @staticmethod
    def _find_blocks(model: torch.nn.Module) -> list[torch.nn.Module]:
        return list(model.transformer.h)

    @staticmethod
    def _hidden_size(model: torch.nn.Module) -> int:
        return int(model.config.n_embd)

    @staticmethod
    def _q_heads(model: torch.nn.Module) -> int:
        return int(model.config.n_head)

    @staticmethod
    def _kv_heads(model: torch.nn.Module) -> int:
        return int(model.config.n_head)

    def register_hooks(self, context: dict[str, Any], handles: list[Any]) -> None:
        for index, block in enumerate(self.blocks):
            context["layer_index"] = index
            attention = block.attn.c_attn

            def qkv_hook(_module: torch.nn.Module, inputs: tuple[Any, ...], output: Any, index: int = index) -> None:
                capture = context["current_layers"][index]
                capture["attn_in"] = _cpu_float(inputs[0])
                capture["attn_raw"] = _cpu_float(output)

            handles.append(attention.register_forward_hook(qkv_hook))
            _capture_input_output(block.attn.c_proj, context, "o", handles, index)
            _capture_input_output(block.mlp.c_fc, context, "fc", handles, index)
            _capture_input_output(block.mlp.c_proj, context, "proj", handles, index)

    def weights(self, block: torch.nn.Module) -> dict[str, torch.Tensor]:
        fused = _module_weight(block.attn.c_attn, transpose=True)
        hidden = self.hidden_size
        return {
            "q": fused[:hidden].clone(),
            "k": fused[hidden : 2 * hidden].clone(),
            "v": fused[2 * hidden :].clone(),
            "o": _module_weight(block.attn.c_proj, transpose=True),
            "fc": _module_weight(block.mlp.c_fc, transpose=True),
            "proj": _module_weight(block.mlp.c_proj, transpose=True),
        }

    def finalize(
        self, captured: dict[str, torch.Tensor], rope: Any
    ) -> tuple[dict[str, torch.Tensor], tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        del rope
        raw = captured["attn_raw"].reshape(-1, 3 * self.hidden_size)
        q, k, v = raw.chunk(3, dim=-1)
        activations = {
            "q": captured["attn_in"].reshape(-1, captured["attn_in"].shape[-1]).contiguous(),
            "k": captured["attn_in"].reshape(-1, captured["attn_in"].shape[-1]).contiguous(),
            "v": captured["attn_in"].reshape(-1, captured["attn_in"].shape[-1]).contiguous(),
            "o": _module_input(captured, "o"),
            "fc": _module_input(captured, "fc"),
            "proj": _module_input(captured, "proj"),
        }
        return activations, (q, k, v)


class OPTAdapter(ModelAdapter):
    @staticmethod
    def _find_blocks(model: torch.nn.Module) -> list[torch.nn.Module]:
        return list(model.model.decoder.layers)

    @staticmethod
    def _hidden_size(model: torch.nn.Module) -> int:
        return int(model.config.hidden_size)

    @staticmethod
    def _q_heads(model: torch.nn.Module) -> int:
        return int(model.config.num_attention_heads)

    @staticmethod
    def _kv_heads(model: torch.nn.Module) -> int:
        return int(model.config.num_attention_heads)

    def register_hooks(self, context: dict[str, Any], handles: list[Any]) -> None:
        for index, block in enumerate(self.blocks):
            context["layer_index"] = index
            attention = block.self_attn
            for name, module in (
                ("q", attention.q_proj),
                ("k", attention.k_proj),
                ("v", attention.v_proj),
                ("o", attention.out_proj),
                ("fc", block.fc1),
                ("proj", block.fc2),
            ):
                _capture_input_output(
                    module, context, name, handles, index, name in {"q", "k", "v"}
                )

    def weights(self, block: torch.nn.Module) -> dict[str, torch.Tensor]:
        attention = block.self_attn
        return {
            "q": _module_weight(attention.q_proj),
            "k": _module_weight(attention.k_proj),
            "v": _module_weight(attention.v_proj),
            "o": _module_weight(attention.out_proj),
            "fc": _module_weight(block.fc1),
            "proj": _module_weight(block.fc2),
        }

    def finalize(
        self, captured: dict[str, torch.Tensor], rope: Any
    ) -> tuple[dict[str, torch.Tensor], tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        del rope
        activations = {
            name: _module_input(captured, name)
            for name in self.roles
        }
        q = _module_output(captured, "q")
        k = _module_output(captured, "k")
        v = _module_output(captured, "v")
        return activations, (q, k, v)


def _apply_rotary(
    q: torch.Tensor,
    k: torch.Tensor,
    rope: Any,
    family: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    if rope is None or not isinstance(rope, (tuple, list)) or len(rope) != 2:
        raise RuntimeError(f"{family} forward did not expose rotary position embeddings")
    cos, sin = rope
    if family == "gpt_neox":
        from transformers.models.gpt_neox.modeling_gpt_neox import apply_rotary_pos_emb
    elif family == "qwen2":
        from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb
    else:  # pragma: no cover - guarded by adapters
        raise ValueError(f"unsupported rotary family: {family}")
    return apply_rotary_pos_emb(q, k, cos, sin)


class GPTNeoXAdapter(ModelAdapter):
    @staticmethod
    def _find_blocks(model: torch.nn.Module) -> list[torch.nn.Module]:
        return list(model.gpt_neox.layers)

    @staticmethod
    def _hidden_size(model: torch.nn.Module) -> int:
        return int(model.config.hidden_size)

    @staticmethod
    def _q_heads(model: torch.nn.Module) -> int:
        return int(model.config.num_attention_heads)

    @staticmethod
    def _kv_heads(model: torch.nn.Module) -> int:
        return int(model.config.num_attention_heads)

    def register_rope_hook(self, context: dict[str, Any], handles: list[Any]) -> None:
        def rope_hook(_module: torch.nn.Module, _inputs: tuple[Any, ...], output: Any) -> None:
            if not isinstance(output, (tuple, list)) or len(output) != 2:
                raise RuntimeError("GPT-NeoX rotary hook returned an unexpected value")
            context["rope"] = (_cpu_float(output[0]), _cpu_float(output[1]))

        handles.append(self.model.gpt_neox.rotary_emb.register_forward_hook(rope_hook))

    def register_hooks(self, context: dict[str, Any], handles: list[Any]) -> None:
        for index, block in enumerate(self.blocks):
            attention = block.attention

            def qkv_hook(_module: torch.nn.Module, inputs: tuple[Any, ...], output: Any, index: int = index) -> None:
                capture = context["current_layers"][index]
                capture["attn_in"] = _cpu_float(inputs[0])
                capture["attn_raw"] = _cpu_float(output)

            handles.append(attention.query_key_value.register_forward_hook(qkv_hook))
            _capture_input_output(attention.dense, context, "o", handles, index)
            _capture_input_output(
                block.mlp.dense_h_to_4h, context, "fc", handles, index
            )
            _capture_input_output(
                block.mlp.dense_4h_to_h, context, "proj", handles, index
            )

    def weights(self, block: torch.nn.Module) -> dict[str, torch.Tensor]:
        fused = _module_weight(block.attention.query_key_value)
        hidden = self.hidden_size
        # GPT-NeoX emits [head, q/k/v, head_dim] after viewing the fused
        # projection.  Store each projection in the same [out, in] convention
        # as the other adapters.
        heads = self.q_heads
        fused = fused.reshape(heads, 3, self.head_dim, hidden)
        q = fused[:, 0].reshape(hidden, hidden)
        k = fused[:, 1].reshape(hidden, hidden)
        v = fused[:, 2].reshape(hidden, hidden)
        return {
            "q": q.contiguous(),
            "k": k.contiguous(),
            "v": v.contiguous(),
            "o": _module_weight(block.attention.dense),
            "fc": _module_weight(block.mlp.dense_h_to_4h),
            "proj": _module_weight(block.mlp.dense_4h_to_h),
        }

    def finalize(
        self, captured: dict[str, torch.Tensor], rope: Any
    ) -> tuple[dict[str, torch.Tensor], tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        raw = captured["attn_raw"].reshape(1, -1, self.q_heads, 3 * self.head_dim).transpose(1, 2)
        q, k, v = raw.chunk(3, dim=-1)
        q, k = _apply_rotary(q, k, rope, "gpt_neox")
        q = q.transpose(1, 2).reshape(-1, self.q_heads * self.head_dim).contiguous()
        k = k.transpose(1, 2).reshape(-1, self.kv_heads * self.head_dim).contiguous()
        v = v.transpose(1, 2).reshape(-1, self.kv_heads * self.head_dim).contiguous()
        activations = {
            "q": captured["attn_in"].reshape(-1, captured["attn_in"].shape[-1]).contiguous(),
            "k": captured["attn_in"].reshape(-1, captured["attn_in"].shape[-1]).contiguous(),
            "v": captured["attn_in"].reshape(-1, captured["attn_in"].shape[-1]).contiguous(),
            "o": _module_input(captured, "o"),
            "fc": _module_input(captured, "fc"),
            "proj": _module_input(captured, "proj"),
        }
        return activations, (q, k, v)


class Qwen2Adapter(ModelAdapter):
    roles = ("q", "k", "v", "o", "fc_gate", "fc_up", "proj")
    role_groups = {
        "q": "q",
        "k": "k",
        "v": "v",
        "o": "o",
        "fc_gate": "fc",
        "fc_up": "fc",
        "proj": "proj",
    }

    @staticmethod
    def _find_blocks(model: torch.nn.Module) -> list[torch.nn.Module]:
        return list(model.model.layers)

    @staticmethod
    def _hidden_size(model: torch.nn.Module) -> int:
        return int(model.config.hidden_size)

    @staticmethod
    def _q_heads(model: torch.nn.Module) -> int:
        return int(model.config.num_attention_heads)

    @staticmethod
    def _kv_heads(model: torch.nn.Module) -> int:
        return int(model.config.num_key_value_heads)

    def register_rope_hook(self, context: dict[str, Any], handles: list[Any]) -> None:
        def rope_hook(_module: torch.nn.Module, _inputs: tuple[Any, ...], output: Any) -> None:
            if not isinstance(output, (tuple, list)) or len(output) != 2:
                raise RuntimeError("Qwen rotary hook returned an unexpected value")
            context["rope"] = (_cpu_float(output[0]), _cpu_float(output[1]))

        handles.append(self.model.model.rotary_emb.register_forward_hook(rope_hook))

    def register_hooks(self, context: dict[str, Any], handles: list[Any]) -> None:
        for index, block in enumerate(self.blocks):
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
                _capture_input_output(
                    module, context, name, handles, index, name in {"q", "k", "v"}
                )

    def weights(self, block: torch.nn.Module) -> dict[str, torch.Tensor]:
        attention = block.self_attn
        return {
            "q": _module_weight(attention.q_proj),
            "k": _module_weight(attention.k_proj),
            "v": _module_weight(attention.v_proj),
            "o": _module_weight(attention.o_proj),
            "fc_gate": _module_weight(block.mlp.gate_proj),
            "fc_up": _module_weight(block.mlp.up_proj),
            "proj": _module_weight(block.mlp.down_proj),
        }

    def finalize(
        self, captured: dict[str, torch.Tensor], rope: Any
    ) -> tuple[dict[str, torch.Tensor], tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        q_raw = _module_output(captured, "q").reshape(1, -1, self.q_heads, self.head_dim).transpose(1, 2)
        k_raw = _module_output(captured, "k").reshape(1, -1, self.kv_heads, self.head_dim).transpose(1, 2)
        v_raw = _module_output(captured, "v").reshape(1, -1, self.kv_heads, self.head_dim).transpose(1, 2)
        q, k = _apply_rotary(q_raw, k_raw, rope, "qwen2")
        q = q.transpose(1, 2).reshape(-1, self.q_heads * self.head_dim).contiguous()
        k = k.transpose(1, 2).reshape(-1, self.kv_heads * self.head_dim).contiguous()
        v = v_raw.transpose(1, 2).reshape(-1, self.kv_heads * self.head_dim).contiguous()
        activations = {name: _module_input(captured, name) for name in self.roles}
        return activations, (q, k, v)


def make_adapter(spec: ModelSpec, model: torch.nn.Module, layers: int | None) -> ModelAdapter:
    adapters = {
        "gpt2": GPT2Adapter,
        "opt": OPTAdapter,
        "gpt_neox": GPTNeoXAdapter,
        "qwen2": Qwen2Adapter,
    }
    try:
        adapter_type = adapters[spec.family]
    except KeyError as exc:  # pragma: no cover - manifest guard
        raise ValueError(f"no adapter for model family {spec.family}") from exc
    return adapter_type(model, layers)


def _load_model(spec: ModelSpec, device: torch.device) -> tuple[Any, torch.nn.Module]:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not spec.path.is_dir():
        raise FileNotFoundError(f"model directory does not exist: {spec.path}")
    tokenizer = AutoTokenizer.from_pretrained(
        spec.path, local_files_only=True, use_fast=True
    )
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        spec.path,
        local_files_only=True,
        torch_dtype=dtype,
    )
    model.eval().to(device)
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False
    return tokenizer, model


def _capture_windows(
    model: torch.nn.Module,
    adapter: ModelAdapter,
    windows: Sequence[Window],
    device: torch.device,
) -> tuple[dict[str, list[list[torch.Tensor]]], list[list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]]]:
    context: dict[str, Any] = {"current_layers": [], "layer_index": 0, "rope": None}
    handles: list[Any] = []
    adapter.register_hooks(context, handles)
    adapter.register_rope_hook(context, handles)
    activation_store = {role: [] for role in adapter.roles}
    qkv_store: list[list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]] = []
    try:
        for window in windows:
            context["current_layers"] = [{} for _ in adapter.blocks]
            context["rope"] = None
            input_ids = torch.tensor(window.input_ids, dtype=torch.long, device=device).unsqueeze(0)
            with torch.no_grad():
                output = model(input_ids=input_ids, use_cache=False)
            if output is None:
                raise RuntimeError("model forward returned no output")
            per_layer_qkv = []
            per_layer_activations = {role: [] for role in adapter.roles}
            for index, captured in enumerate(context["current_layers"]):
                context["layer_index"] = index
                activations, qkv = adapter.finalize(captured, context["rope"])
                for role in adapter.roles:
                    per_layer_activations[role].append(activations[role])
                per_layer_qkv.append(qkv)
            for role in adapter.roles:
                activation_store[role].append(per_layer_activations[role])
            qkv_store.append(per_layer_qkv)
    finally:
        for handle in handles:
            handle.remove()
    return activation_store, qkv_store


def collect_model_data(
    spec: ModelSpec,
    data_dir: Path,
    sequence_length: int,
    calibration_samples: int,
    test_samples: int,
    device_name: str,
    layers: int | None,
) -> ModelData:
    device = torch.device(device_name)
    tokenizer, model = _load_model(spec, device)
    try:
        calibration_windows, test_windows, data_metadata = load_real_windows(
            tokenizer,
            data_dir,
            sequence_length,
            calibration_samples,
            test_samples,
        )
        adapter = make_adapter(spec, model, layers)
        weights = [adapter.weights(block) for block in adapter.blocks]
        calibration_activations, calibration_qkv = _capture_windows(
            model, adapter, calibration_windows, device
        )
        test_activations, test_qkv = _capture_windows(
            model, adapter, test_windows, device
        )
        config = model.config
        metadata = {
            "model": spec.name,
            "family": spec.family,
            "source_revision": spec.source_revision,
            "local_path": str(spec.path),
            "model_type": str(getattr(config, "model_type", "unknown")),
            "architecture": list(getattr(config, "architectures", []) or []),
            "parameter_dtype": str(next(model.parameters()).dtype),
            "device": str(device),
            "layers": len(adapter.blocks),
            "hidden_size": adapter.hidden_size,
            "q_heads": adapter.q_heads,
            "kv_heads": adapter.kv_heads,
            "head_dim": adapter.head_dim,
            "roles": list(adapter.roles),
            "role_groups": adapter.role_groups,
            "requested_layers": layers,
            "data": data_metadata,
        }
        metadata["data"] = dict(data_metadata)
        metadata["data"].update(
            {
                "sequence_length": sequence_length,
                "calibration_samples": calibration_samples,
                "test_samples": test_samples,
            }
        )
        return ModelData(
            spec=spec,
            tokenizer_name=type(tokenizer).__name__,
            layers=len(adapter.blocks),
            hidden_size=adapter.hidden_size,
            q_heads=adapter.q_heads,
            kv_heads=adapter.kv_heads,
            head_dim=adapter.head_dim,
            roles=adapter.roles,
            role_groups=dict(adapter.role_groups),
            weights=weights,
            calibration_activations=calibration_activations,
            test_activations=test_activations,
            calibration_qkv=calibration_qkv,
            test_qkv=test_qkv,
            calibration_windows=calibration_windows,
            test_windows=test_windows,
            metadata=metadata,
        )
    finally:
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()


def load_or_collect_model_data(
    spec: ModelSpec,
    data_dir: Path,
    cache_dir: Path,
    sequence_length: int,
    calibration_samples: int,
    test_samples: int,
    device_name: str,
    requested_layers: int | None,
    cache_mode: str,
) -> tuple[ModelData, str, Path]:
    """Load a validated snapshot or capture the model according to ``cache_mode``.

    ``read`` is intentionally fail-closed: a missing or stale cache never falls
    back to a model load.  ``auto`` may create a missing/stale cache; ``write``
    always refreshes it; ``off`` keeps the historical one-shot behavior.
    """

    if cache_mode not in {"auto", "read", "write", "off"}:
        raise ValueError(f"unknown cache mode: {cache_mode}")
    path = model_cache_path(
        spec,
        cache_dir,
        sequence_length,
        calibration_samples,
        test_samples,
        requested_layers,
    )
    if cache_mode in {"auto", "read"} and path.is_file():
        try:
            return (
                load_model_cache(
                    path,
                    spec,
                    sequence_length,
                    calibration_samples,
                    test_samples,
                    requested_layers,
                ),
                "cache",
                path,
            )
        except Exception as exc:
            if cache_mode == "read":
                raise CacheValidationError(
                    f"cache-only mode refused invalid cache {path}: {exc}"
                ) from exc
            print(f"{spec.name}: cache invalid, recapturing ({exc})", flush=True)
    elif cache_mode == "read":
        raise FileNotFoundError(f"cache-only mode requires an existing cache: {path}")

    data = collect_model_data(
        spec,
        data_dir,
        sequence_length,
        calibration_samples,
        test_samples,
        device_name,
        requested_layers,
    )
    if cache_mode in {"auto", "write"}:
        save_model_cache(data, path, requested_layers)
    return data, "model_forward", path


def _linear_detail(
    solution: Any,
    weight_pair: tuple[torch.Tensor, torch.Tensor],
    activation_pairs: Sequence[tuple[torch.Tensor, torch.Tensor]],
    activation_state: Any,
    weight_params: Any,
) -> dict[str, Any]:
    """Score a Linear case against the frozen evaluator-side reference."""

    weight_reference = dequantize_nvfp4(*weight_pair).to(torch.float32)
    weight_standard = std_hif4(solution, weight_reference)
    weight_player = dequantize_hif4(weight_params, weight_reference.shape)
    standard_sum = 0.0
    player_sum = 0.0
    relative: list[float] = []
    elements = 0
    for activation_pair in activation_pairs:
        activation_reference = dequantize_nvfp4(*activation_pair).to(torch.float32)
        reference = activation_reference @ weight_reference.T
        standard = std_hif4(solution, activation_reference) @ weight_standard.T
        player_params = solution.hif4_dynamic_quantize_activation(
            *activation_pair, activation_state
        )
        player_activation = dequantize_hif4(
            player_params, activation_reference.shape
        )
        player = player_activation @ weight_player.T
        standard_mse = float((standard - reference).square().mean())
        player_mse = float((player - reference).square().mean())
        standard_sum += standard_mse * reference.numel()
        player_sum += player_mse * reference.numel()
        elements += int(reference.numel())
        if standard_mse <= 0.0:
            raise ZeroDivisionError("official case has non-positive MSE_STD")
        relative.append((standard_mse - player_mse) / standard_mse)
    return {
        "gain": sum(relative) / len(relative),
        "score_sum": sum(relative),
        "case_count": len(relative),
        "case_scores": relative,
        "standard_sum": standard_sum,
        "player_sum": player_sum,
        "elements": elements,
    }


def _attention_detail(
    solution: Any,
    qkv_pairs: Sequence[
        tuple[
            tuple[torch.Tensor, torch.Tensor],
            tuple[torch.Tensor, torch.Tensor],
            tuple[torch.Tensor, torch.Tensor],
        ]
    ],
    q_state: Any,
    k_state: Any,
    v_state: Any,
    q_heads: int,
    kv_heads: int,
    head_dim: int,
) -> dict[str, Any]:
    relative: list[float] = []
    standard_sum = 0.0
    player_sum = 0.0
    elements = 0
    for q_pair, k_pair, v_pair in qkv_pairs:
        q_reference = dequantize_nvfp4(*q_pair).to(torch.float32)
        k_reference = dequantize_nvfp4(*k_pair).to(torch.float32)
        v_reference = dequantize_nvfp4(*v_pair).to(torch.float32)
        q_standard = std_hif4(solution, q_reference)
        k_standard = std_hif4(solution, k_reference)
        v_standard = std_hif4(solution, v_reference)
        q_params = solution.hif4_dynamic_quantize_q(
            *q_pair, q_heads, head_dim, q_state
        )
        k_params = solution.hif4_dynamic_quantize_k(
            *k_pair, kv_heads, head_dim, k_state
        )
        v_params = solution.hif4_dynamic_quantize_v(
            *v_pair, kv_heads, head_dim, v_state
        )
        q_player = dequantize_hif4(q_params, q_reference.shape)
        k_player = dequantize_hif4(k_params, k_reference.shape)
        v_player = dequantize_hif4(v_params, v_reference.shape)
        reference = causal_attention(
            q_reference[None], k_reference[None], v_reference[None],
            q_heads, kv_heads, head_dim, False
        )
        standard = causal_attention(
            q_standard[None], k_standard[None], v_standard[None],
            q_heads, kv_heads, head_dim, False
        )
        player = causal_attention(
            q_player[None], k_player[None], v_player[None],
            q_heads, kv_heads, head_dim, False
        )
        standard_mse = float((standard - reference).square().mean())
        player_mse = float((player - reference).square().mean())
        standard_sum += standard_mse * reference.numel()
        player_sum += player_mse * reference.numel()
        elements += int(reference.numel())
        if standard_mse <= 0.0:
            raise ZeroDivisionError("official case has non-positive MSE_STD")
        relative.append((standard_mse - player_mse) / standard_mse)
    return {
        "gain": sum(relative) / len(relative),
        "score_sum": sum(relative),
        "case_count": len(relative),
        "case_scores": relative,
        "standard_sum": standard_sum,
        "player_sum": player_sum,
        "elements": elements,
    }


def _aggregate_details(details: Sequence[dict[str, Any]]) -> dict[str, float]:
    if not details:
        return {
            "official_score_sum": 0.0,
            "official_score_mean": float("nan"),
            "official_case_count": 0,
            "macro_gain": float("nan"),
            "global_gain": float("nan"),
            "cases": 0,
        }
    standard = sum(item["standard_sum"] for item in details)
    player = sum(item["player_sum"] for item in details)
    official_score_sum = sum(item["score_sum"] for item in details)
    official_case_count = sum(int(item["case_count"]) for item in details)
    return {
        "official_score_sum": official_score_sum,
        "official_score_mean": official_score_sum / official_case_count,
        "official_case_count": official_case_count,
        "macro_gain": sum(item["gain"] for item in details) / len(details),
        "global_gain": (standard - player) / max(standard, 1.0e-30),
        "standard_sum": standard,
        "player_sum": player,
        "elements": sum(int(item["elements"]) for item in details),
        "cases": len(details),
    }


def build_panel_score(
    official_flow_score: dict[str, Any],
    panel_profile: str = DEFAULT_PANEL_PROFILE,
) -> dict[str, Any]:
    """Project native case means onto a fixed reference panel.

    The local capture cannot reproduce the hidden official 250/200 examples,
    and the available Qwen snapshot has a different number of layer/role
    cases.  Repeating cases would make the score artificially optimistic, so
    the shaped panel preserves each component's native mean and only applies
    the official component counts.  ``native`` remains available for
    backwards-compatible diagnostics.
    """

    if panel_profile not in PANEL_PROFILES:
        raise ValueError(
            f"unknown panel profile {panel_profile!r}; choose from {PANEL_PROFILES}"
        )
    linear = float(official_flow_score.get("linear", 0.0))
    attention = float(official_flow_score.get("attention", 0.0))
    # Very old hand-built ranking fixtures omitted case counts.  Treat a
    # present component as one aggregate case solely for backwards-compatible
    # diagnostics; real evaluator results always persist explicit counts.
    linear_cases_inferred = "linear_cases" not in official_flow_score
    attention_cases_inferred = "attention_cases" not in official_flow_score
    linear_cases = int(
        official_flow_score.get("linear_cases", 1 if "linear" in official_flow_score else 0)
    )
    attention_cases = int(
        official_flow_score.get(
            "attention_cases", 1 if "attention" in official_flow_score else 0
        )
    )
    total_cases = linear_cases + attention_cases
    if panel_profile == "native":
        return {
            "profile": "native",
            "revision": None,
            "linear": linear,
            "attention": attention,
            "total": linear + attention,
            "linear_cases": linear_cases,
            "attention_cases": attention_cases,
            "total_cases": total_cases,
            "source_linear_cases": linear_cases,
            "source_attention_cases": attention_cases,
            "source_case_counts_inferred": (
                linear_cases_inferred or attention_cases_inferred
            ),
            "linear_mean": linear / linear_cases if linear_cases else float("nan"),
            "attention_mean": (
                attention / attention_cases if attention_cases else float("nan")
            ),
            "aggregation": "native_sum",
        }

    linear_mean = linear / linear_cases if linear_cases else float("nan")
    attention_mean = attention / attention_cases if attention_cases else float("nan")
    shaped_linear = REFERENCE_PANEL_LINEAR_CASES * linear_mean
    shaped_attention = REFERENCE_PANEL_ATTENTION_CASES * attention_mean
    return {
        "profile": "qwen-official",
        "revision": OFFICIAL_PANEL_REVISION,
        "linear": shaped_linear,
        "attention": shaped_attention,
        "total": shaped_linear + shaped_attention,
        "linear_cases": REFERENCE_PANEL_LINEAR_CASES,
        "attention_cases": REFERENCE_PANEL_ATTENTION_CASES,
        "total_cases": REFERENCE_PANEL_TOTAL_CASES,
        "source_linear_cases": linear_cases,
        "source_attention_cases": attention_cases,
        "source_case_counts_inferred": (
            linear_cases_inferred or attention_cases_inferred
        ),
        "linear_mean": linear_mean,
        "attention_mean": attention_mean,
        "aggregation": "source_component_mean_times_fixed_panel_count",
    }


def _result_panel_score(
    result: dict[str, Any], panel_profile: str
) -> dict[str, Any]:
    """Read a persisted panel score or derive it for old result JSON."""

    panel = result.get("panel_score")
    if isinstance(panel, dict) and panel.get("profile") == panel_profile:
        return panel
    return build_panel_score(result["official_flow_score"], panel_profile)


def evaluate_candidate(
    candidate: CandidateSpec,
    data: ModelData,
    mode: str,
    algorithm_device_name: str,
    panel_profile: str = DEFAULT_PANEL_PROFILE,
) -> dict[str, Any]:
    if not candidate.path.is_file():
        raise FileNotFoundError(f"candidate source does not exist: {candidate.path}")
    solution = load_solution(candidate.path)
    stats = instrument_solution(solution)
    algorithm_device = torch.device(algorithm_device_name)
    linear_cases: dict[str, list[dict[str, float]]] = {role: [] for role in data.roles}
    attention_cases: list[dict[str, float]] = []
    if algorithm_device.type == "cuda":
        torch.cuda.synchronize(algorithm_device)
    started = time.perf_counter()
    for layer_index in range(data.layers):
        for role in data.roles:
            weight_pair = nvfp4_encode(
                data.weights[layer_index][role].to(algorithm_device), mode
            )
            calibration_pairs = [
                nvfp4_encode(
                    data.calibration_activations[role][batch][layer_index].to(
                        algorithm_device
                    ),
                    mode,
                )
                for batch in range(len(data.calibration_windows))
            ]
            calibrated = solution.hif4_calibration_and_quantize_weight(
                *weight_pair, calibration_pairs
            )
            if not isinstance(calibrated, dict) or not {
                "weight_params",
                "activation_state",
            }.issubset(calibrated):
                raise ValueError(
                    "hif4_calibration_and_quantize_weight must return "
                    "weight_params and activation_state"
                )
            validate_state(calibrated["activation_state"])
            test_pairs = [
                nvfp4_encode(
                    data.test_activations[role][batch][layer_index].to(
                        algorithm_device
                    ),
                    mode,
                )
                for batch in range(len(data.test_windows))
            ]
            linear_cases[role].append(
                _linear_detail(
                    solution,
                    weight_pair,
                    test_pairs,
                    calibrated["activation_state"],
                    calibrated["weight_params"],
                )
            )

        calibration_qkv = []
        for batch in range(len(data.calibration_windows)):
            q, k, v = data.calibration_qkv[batch][layer_index]
            calibration_qkv.append(
                {
                    "q": nvfp4_encode(q.to(algorithm_device), mode),
                    "k": nvfp4_encode(k.to(algorithm_device), mode),
                    "v": nvfp4_encode(v.to(algorithm_device), mode),
                }
            )
        states = solution.hif4_calibration_attention(
            calibration_qkv, data.q_heads, data.kv_heads, data.head_dim
        )
        if not isinstance(states, dict) or not {
            "q_state",
            "k_state",
            "v_state",
        }.issubset(states):
            raise ValueError(
                "hif4_calibration_attention must return q_state, k_state, and v_state"
            )
        validate_state(states["q_state"])
        validate_state(states["k_state"])
        validate_state(states["v_state"])
        test_qkv = []
        for batch in range(len(data.test_windows)):
            q, k, v = data.test_qkv[batch][layer_index]
            test_qkv.append(
                (
                    nvfp4_encode(q.to(algorithm_device), mode),
                    nvfp4_encode(k.to(algorithm_device), mode),
                    nvfp4_encode(v.to(algorithm_device), mode),
                )
            )
        attention_cases.append(
            _attention_detail(
                solution,
                test_qkv,
                states["q_state"],
                states["k_state"],
                states["v_state"],
                data.q_heads,
                data.kv_heads,
                data.head_dim,
            )
        )
    if algorithm_device.type == "cuda":
        torch.cuda.synchronize(algorithm_device)
    elapsed = time.perf_counter() - started

    linear_by_role = {
        role: _aggregate_details(details) for role, details in linear_cases.items()
    }
    linear_by_group: dict[str, list[dict[str, float]]] = {}
    for role, details in linear_cases.items():
        linear_by_group.setdefault(data.role_groups[role], []).extend(details)
    linear_groups = {
        group: _aggregate_details(details) for group, details in linear_by_group.items()
    }
    linear_all = _aggregate_details(
        [item for details in linear_cases.values() for item in details]
    )
    component_macro = sum(item["global_gain"] for item in linear_groups.values()) / len(linear_groups)
    attention = _aggregate_details(attention_cases)
    official_score = {
        "linear": linear_all["official_score_sum"],
        "attention": attention["official_score_sum"],
        "total": (
            linear_all["official_score_sum"] + attention["official_score_sum"]
        ),
        "linear_cases": linear_all["official_case_count"],
        "attention_cases": attention["official_case_count"],
        "total_cases": (
            linear_all["official_case_count"] + attention["official_case_count"]
        ),
    }
    panel_score = build_panel_score(official_score, panel_profile)
    api_total = stats["calibration"] + stats["dynamic"]
    return {
        "candidate": candidate.name,
        "source": str(candidate.path),
        "source_sha256": sha256_file(candidate.path),
        "official_score": candidate.official_score,
        "official_time": candidate.official_time,
        "model": data.spec.name,
        "linear_by_role": linear_by_role,
        "linear_by_group": linear_groups,
        "linear": linear_all,
        "linear_component_macro_gain": component_macro,
        "attention": attention,
        "official_flow_score": official_score,
        "panel_score": panel_score,
        "timing": {
            "wall_seconds": elapsed,
            "algorithm_stage_seconds": (
                None
                if stats["first_start"] is None
                else stats["last_end"] - stats["first_start"]
            ),
            "calibration_seconds": stats["calibration"],
            "dynamic_seconds": stats["dynamic"],
            "official_api_total_seconds": api_total,
            "api_calls": stats["calls"],
            "nested_api_calls": stats["nested_calls"],
            "under_official_runtime_limit": (
                api_total < OFFICIAL_RUNTIME_LIMIT_SECONDS
            ),
            # Kept only as a compatibility diagnostic for pre-revision JSON;
            # it must not decide current submission validity.
            "under_300_seconds": api_total < 300.0,
        },
    }


def _mean(values: Iterable[float]) -> float:
    values = list(values)
    return sum(values) / len(values) if values else float("nan")


def pearson(xs: Sequence[float], ys: Sequence[float]) -> float:
    if len(xs) != len(ys) or len(xs) < 2:
        return float("nan")
    x_mean = _mean(xs)
    y_mean = _mean(ys)
    numerator = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys))
    denominator = math.sqrt(
        sum((x - x_mean) ** 2 for x in xs) * sum((y - y_mean) ** 2 for y in ys)
    )
    return numerator / denominator if denominator else float("nan")


def _average_ranks(values: Sequence[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda index: values[index])
    ranks = [0.0] * len(values)
    cursor = 0
    while cursor < len(order):
        end = cursor + 1
        while end < len(order) and values[order[end]] == values[order[cursor]]:
            end += 1
        rank = (cursor + end - 1) / 2.0 + 1.0
        for index in order[cursor:end]:
            ranks[index] = rank
        cursor = end
    return ranks


def spearman(xs: Sequence[float], ys: Sequence[float]) -> float:
    return pearson(_average_ranks(xs), _average_ranks(ys))


def _pairwise_rank_agreement(xs: Sequence[float], ys: Sequence[float]) -> float:
    if len(xs) < 2:
        return float("nan")
    total = 0
    agreed = 0
    for left in range(len(xs)):
        for right in range(left + 1, len(xs)):
            dx = xs[left] - xs[right]
            dy = ys[left] - ys[right]
            if dx == 0 or dy == 0:
                continue
            total += 1
            if (dx > 0) == (dy > 0):
                agreed += 1
    return agreed / total if total else float("nan")


def audit_official_ranking(
    results: Sequence[dict[str, Any]],
    requested_candidates: Sequence[str],
    candidate_specs: dict[str, CandidateSpec] | None = None,
    expected_models: Sequence[str] | None = None,
    panel_profile: str = DEFAULT_PANEL_PROFILE,
    primary_model: str = DEFAULT_PRIMARY_MODEL,
    guardrail_models: Sequence[str] | None = None,
) -> dict[str, Any]:
    if panel_profile not in PANEL_PROFILES:
        raise ValueError(
            f"unknown panel profile {panel_profile!r}; choose from {PANEL_PROFILES}"
        )
    candidate_specs = CANDIDATE_SPECS if candidate_specs is None else candidate_specs
    by_model: dict[str, dict[str, dict[str, Any]]] = {}
    for result in results:
        by_model.setdefault(result["model"], {})[result["candidate"]] = result
    official = {
        name: candidate_specs[name].official_score
        for name in requested_candidates
        if name in candidate_specs and candidate_specs[name].official_score is not None
        and candidate_specs[name].official_panel_revision == OFFICIAL_PANEL_REVISION
    }
    model_features: dict[str, dict[str, dict[str, float]]] = {}
    for model_name, candidate_results in by_model.items():
        model_features[model_name] = {}
        for feature_name, getter in (
            ("official_flow_total", lambda item: item["official_flow_score"]["total"]),
            ("official_flow_linear", lambda item: item["official_flow_score"]["linear"]),
            ("official_flow_attention", lambda item: item["official_flow_score"]["attention"]),
            ("linear_global_gain", lambda item: item["linear"]["global_gain"]),
            ("linear_macro_gain", lambda item: item["linear"]["macro_gain"]),
            ("component_macro_gain", lambda item: item["linear_component_macro_gain"]),
            ("attention_global_gain", lambda item: item["attention"]["global_gain"]),
            ("attention_macro_gain", lambda item: item["attention"]["macro_gain"]),
            (
                "panel_score_linear",
                lambda item: _result_panel_score(item, panel_profile)["linear"],
            ),
            (
                "panel_score_attention",
                lambda item: _result_panel_score(item, panel_profile)["attention"],
            ),
            (
                "panel_score_total",
                lambda item: _result_panel_score(item, panel_profile)["total"],
            ),
        ):
            model_features[model_name][feature_name] = {
                candidate: float(getter(candidate_results[candidate]))
                for candidate in requested_candidates
                if candidate in candidate_results
                and "error" not in candidate_results[candidate]
            }

    model_names = list(expected_models) if expected_models is not None else list(by_model)
    feature_model_names = [name for name in model_names if name in model_features]
    feature_model_names.extend(
        name for name in model_features if name not in feature_model_names
    )
    if primary_model in model_features:
        effective_primary_model = primary_model
    else:
        effective_primary_model = next(iter(feature_model_names), None)
    primary_model_fallback = bool(
        effective_primary_model is not None and effective_primary_model != primary_model
    )
    if guardrail_models is None:
        selected_guardrails = [
            name for name in feature_model_names if name != effective_primary_model
        ]
    else:
        selected_guardrails = [
            name
            for name in guardrail_models
            if name in model_features and name != effective_primary_model
        ]

    aggregate_features: dict[str, dict[str, float]] = {}
    summed_features = {
        "official_flow_total",
        "official_flow_linear",
        "official_flow_attention",
    }
    def aggregate_feature(feature_name: str, candidate: str) -> float:
        values = [
            model_features[model_name][feature_name][candidate]
            for model_name in model_features
            if candidate in model_features[model_name].get(feature_name, {})
        ]
        if not values:
            return float("nan")
        return sum(values) if feature_name in summed_features else _mean(values)

    for feature_name in next(iter(model_features.values()), {}):
        aggregate_features[feature_name] = {
            candidate: aggregate_feature(feature_name, candidate)
            for candidate in requested_candidates
        }

    panel_features = {
        "panel_score_linear",
        "panel_score_attention",
        "panel_score_total",
    }
    for feature_name in panel_features:
        if feature_name not in aggregate_features:
            aggregate_features[feature_name] = {
                candidate: aggregate_feature(feature_name, candidate)
                for candidate in requested_candidates
            }

    def model_feature_value(
        model_name: str | None, feature_name: str, candidate: str
    ) -> float:
        if model_name is None:
            return float("nan")
        return model_features.get(model_name, {}).get(feature_name, {}).get(
            candidate, float("nan")
        )

    def mean_model_feature(
        model_list: Sequence[str], feature_name: str, candidate: str
    ) -> float:
        values = [
            model_features[model_name][feature_name][candidate]
            for model_name in model_list
            if candidate in model_features.get(model_name, {}).get(feature_name, {})
            and math.isfinite(model_features[model_name][feature_name][candidate])
        ]
        return _mean(values)

    for component in ("linear", "attention", "total"):
        feature_name = f"panel_score_{component}"
        aggregate_features[f"primary_panel_score_{component}"] = {
            candidate: model_feature_value(
                effective_primary_model, feature_name, candidate
            )
            for candidate in requested_candidates
        }
        aggregate_features[f"guardrail_panel_mean_{component}"] = {
            candidate: mean_model_feature(
                selected_guardrails, feature_name, candidate
            )
            for candidate in requested_candidates
        }
        aggregate_features[f"all_model_panel_mean_{component}"] = {
            candidate: mean_model_feature(
                feature_model_names, feature_name, candidate
            )
            for candidate in requested_candidates
        }

    expected_model_count = len(model_names)
    candidate_status = {}
    for candidate in requested_candidates:
        candidate_results = [
            by_model[model_name][candidate]
            for model_name in model_names
            if model_name in by_model
            if candidate in by_model[model_name]
            and "error" not in by_model[model_name][candidate]
        ]
        candidate_errors = [
            by_model[model_name][candidate]["error"]
            for model_name in model_names
            if model_name in by_model
            if candidate in by_model[model_name]
            and "error" in by_model[model_name][candidate]
        ]
        api_times = [
            float(item["timing"]["official_api_total_seconds"])
            for item in candidate_results
        ]
        complete = (
            len(candidate_results) == expected_model_count
            and expected_model_count > 0
        )
        all_under_official = bool(api_times) and all(
            value < OFFICIAL_RUNTIME_LIMIT_SECONDS for value in api_times
        )
        primary_result = (
            by_model.get(effective_primary_model, {}).get(candidate)
            if effective_primary_model is not None
            else None
        )
        primary_evaluated = bool(
            primary_result is not None and "error" not in primary_result
        )
        primary_api_time = (
            float(primary_result["timing"]["official_api_total_seconds"])
            if primary_evaluated
            else float("nan")
        )
        primary_panel_total = (
            _result_panel_score(primary_result, panel_profile)["total"]
            if primary_evaluated
            else float("nan")
        )
        primary_panel_finite = math.isfinite(float(primary_panel_total))
        primary_panel_valid = bool(
            primary_evaluated
            and primary_api_time < OFFICIAL_RUNTIME_LIMIT_SECONDS
            and primary_panel_finite
        )
        # The shaped panel is intentionally Qwen-first: missing/slow soft
        # guardrails do not veto a candidate that passes the primary model.
        valid_submission = (
            primary_panel_valid
            if panel_profile != "native"
            else complete and all_under_official
        )
        candidate_status[candidate] = {
            "evaluated_models": len(candidate_results),
            "expected_models": expected_model_count,
            "complete": complete,
            "errors": candidate_errors,
            "official_api_total_seconds": max(api_times) if api_times else float("nan"),
            "proxy_api_seconds_sum": sum(api_times),
            "under_official_runtime_limit": complete and all_under_official,
            "primary_model": effective_primary_model,
            "primary_model_requested": primary_model,
            "primary_model_fallback": primary_model_fallback,
            "primary_evaluated": primary_evaluated,
            "primary_api_total_seconds": primary_api_time,
            "primary_panel_score": primary_panel_total,
            "primary_panel_valid": primary_panel_valid,
            # Compatibility diagnostic; validity follows the revised limit.
            "under_300_seconds": complete and all(
                value < 300.0 for value in api_times
            ),
            "valid_submission": valid_submission,
        }

    ranking_audit: dict[str, Any] = {}
    for feature_name, values in aggregate_features.items():
        names = [candidate for candidate in requested_candidates if candidate in official and math.isfinite(values[candidate])]
        if len(names) < 2:
            continue
        xs = [values[name] for name in names]
        ys = [float(official[name]) for name in names]
        ranking_audit[feature_name] = {
            "candidates": names,
            "local_values": {name: values[name] for name in names},
            "official_scores": {name: official[name] for name in names},
            "pearson": pearson(xs, ys),
            "spearman": spearman(xs, ys),
            "pairwise_rank_agreement": _pairwise_rank_agreement(xs, ys),
        }

    ordering = {}
    if {"c39", "c40"}.issubset(requested_candidates):
        for model_name, values in model_features.items():
            c39 = values.get("official_flow_total", {}).get("c39", float("nan"))
            c40 = values.get("official_flow_total", {}).get("c40", float("nan"))
            ordering[model_name] = {
                "c39_official_flow_total": c39,
                "c40_official_flow_total": c40,
                "c39_above_c40": bool(math.isfinite(c39) and math.isfinite(c40) and c39 > c40),
            }
        aggregate_order = aggregate_features.get("official_flow_total", {})
        ordering["aggregate"] = {
            "c39_official_flow_total": aggregate_order.get("c39", float("nan")),
            "c40_official_flow_total": aggregate_order.get("c40", float("nan")),
            "c39_above_c40": bool(
                math.isfinite(aggregate_order.get("c39", float("nan")))
                and math.isfinite(aggregate_order.get("c40", float("nan")))
                and aggregate_order["c39"] > aggregate_order["c40"]
            ),
        }
    official_flow_values = aggregate_features.get("official_flow_total", {})
    local_order = sorted(
        (
            {"candidate": candidate, "score": official_flow_values[candidate]}
            for candidate in requested_candidates
            if candidate in official_flow_values
            and candidate_status.get(candidate, {}).get("valid_submission", False)
        ),
        key=lambda item: item["score"],
        reverse=True,
    )
    primary_panel_values = aggregate_features.get("primary_panel_score_total", {})
    primary_panel_order = sorted(
        (
            {"candidate": candidate, "score": primary_panel_values[candidate]}
            for candidate in requested_candidates
            if candidate in primary_panel_values
            and math.isfinite(primary_panel_values[candidate])
            and candidate_status.get(candidate, {}).get("valid_submission", False)
        ),
        key=lambda item: item["score"],
        reverse=True,
    )
    return {
        "panel_profile": panel_profile,
        "panel_revision": (
            OFFICIAL_PANEL_REVISION if panel_profile == "qwen-official" else None
        ),
        "reference_panel": {
            "linear_cases": REFERENCE_PANEL_LINEAR_CASES,
            "attention_cases": REFERENCE_PANEL_ATTENTION_CASES,
            "total_cases": REFERENCE_PANEL_TOTAL_CASES,
        },
        "primary_model": effective_primary_model,
        "primary_model_requested": primary_model,
        "primary_model_fallback": primary_model_fallback,
        "guardrail_models": selected_guardrails,
        "official_anchor_scores": official,
        "model_features": model_features,
        "aggregate_features": aggregate_features,
        "candidate_status": candidate_status,
        "local_official_flow_order": local_order,
        "local_primary_panel_order": primary_panel_order,
        "local_panel_order": primary_panel_order,
        "ranking_audit": ranking_audit,
        "c39_vs_c40": ordering,
        "warning": (
            "official anchors audit ranking only; they are never candidate inputs "
            "or regression targets. qwen-official is a mean-preserving local shape, "
            "not an absolute official-score conversion"
        ),
    }


def _json_safe(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    return value


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_json_safe(value), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def write_report(
    path: Path,
    run_metadata: dict[str, Any],
    model_status: Sequence[dict[str, Any]],
    results: Sequence[dict[str, Any]],
    fit: dict[str, Any],
) -> None:
    reference_panel = run_metadata.get("reference_panel", {})
    panel_profile = run_metadata.get(
        "panel_profile", fit.get("panel_profile", DEFAULT_PANEL_PROFILE)
    )
    primary_model = run_metadata.get(
        "primary_model", fit.get("primary_model", DEFAULT_PRIMARY_MODEL)
    )
    primary_warning = run_metadata.get("primary_model_selection_warning")
    lines = [
        "# Qwen 主模型本地评测报告",
        "",
        f"运行时间：{run_metadata['started_at']}（配置 mode={run_metadata['mode']}，seq={run_metadata['sequence_length']}，calib={run_metadata['calibration_samples']}，test={run_metadata['test_samples']}，cache_mode={run_metadata['cache_mode']}）",
        "",
        f"主评测配置：`{panel_profile}`，主模型 `{primary_model}`，参考形状为 {reference_panel.get('linear_cases', REFERENCE_PANEL_LINEAR_CASES)} Linear + {reference_panel.get('attention_cases', REFERENCE_PANEL_ATTENTION_CASES)} Attention。",
        "本地 shaped panel 只把冻结语料上每个组件的平均 case gain 投影到官方样例数量，不复制 case、不拟合官方绝对分数。官方分数没有进入候选校准状态，也没有传给 `solution.py`。评估器内部的输出矩阵乘法只在候选返回量化结果之后，用作固定参考误差；候选离线校准可以自行用 `A@W` 优化 `Q(W)`，但不得将其用于 `Q(A)` 或写入 `activation_state`。",
        "官方上下文：外部 `youxilee/hif4` 用户提供结果为 24153/239s，仅作不可导入的参考；新增 2 个用例呈 Qwen 30B-like 特征，但完整输入尚未公开。",
        "",
        "## 数据与模型完整性",
        "",
        f"- 数据集：`Salesforce/wikitext` / `{WIKITEXT_CONFIG}` / revision `{WIKITEXT_REVISION}`。",
        f"- 评分协议：v{run_metadata['scoring_protocol']['version']}；标准 codec SHA256 `{run_metadata['scoring_protocol']['standard_codec_sha256']}`。",
        "- calibration 来自 train，test 来自 validation；每个窗口来自一个文档，禁止环形重复、窗口重叠和跨 split 文档复用。",
        "- Qwen2.5-0.5B（GQA、RoPE、SwiGLU）承担主排序；其他模型只作为软 guardrail，缺失或轻微回退不会覆盖 Qwen 主分。",
        "- 模型状态：",
        "",
            "| 模型 | 状态 | 层数 | hidden | heads / kv-heads | 数据来源 | 说明 |",
            "|---|---|---:|---:|---:|---|---|",
    ]
    if primary_warning:
        lines.insert(4, f"- 主模型选择提示：{primary_warning}。")
    for status in model_status:
        if status.get("status") == "loaded":
            metadata = status["metadata"]
            lines.append(
                f"| {status['model']} | loaded | {metadata['layers']} | {metadata['hidden_size']} | {metadata['q_heads']} / {metadata['kv_heads']} | {status.get('source', 'model_forward')} | {metadata['family']} |"
            )
        else:
            lines.append(
                f"| {status['model']} | skipped | - | - | - | - | {status.get('error', 'unknown error')} |"
            )

    lines.extend(
        [
            "",
            "## 候选在各模型上的结果",
            "",
            "每个 native case 先计算 `(MSE_STD-MSE_PLAYER)/MSE_STD`。`official_flow_total` 保留原始 case 求和；主排序使用 `panel_score.total = 250*Linear_mean + 200*Attention_mean`，因此不会因模型层数或本地窗口数不同而放大。global-MSE 和组件均值只保留为诊断。",
            "",
            "| 模型 | 候选 | Native total | Panel total | Panel Linear | Panel Attention | Source cases (L/A) | API time(s) |",
            "|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for result in results:
        timing = result["timing"]
        score = result["official_flow_score"]
        panel = _result_panel_score(result, panel_profile)
        lines.append(
            f"| {result['model']} | {result['candidate']} | {score['total']:.6f} | {panel['total']:.6f} | {panel['linear']:.6f} | {panel['attention']:.6f} | {panel['source_linear_cases']}/{panel['source_attention_cases']} | {timing['official_api_total_seconds']:.3f} |"
        )

    official_scores = fit.get("official_anchor_scores", {})
    if official_scores:
        anchor_description = "、".join(
            f"{name.upper()}={score}" for name, score in official_scores.items()
        )
        fit_description = (
            f"官方锚点：{anchor_description}。下表只审计候选排列是否一致，不拟合或预测官方绝对分数。"
        )
    else:
        fit_description = (
            "本次运行评测的是自定义候选，没有把官方分数传入候选或当次拟合；"
            "本报告只输出官方流程代理总分，不执行官方绝对分数回归。"
        )
    lines.extend(["", "## 与官方锚点的排序审计", "", fit_description])
    fitted_features = fit.get("ranking_audit", {})
    if fitted_features:
        lines.extend(
            [
                "",
                "| 本地特征 | Spearman | pairwise rank agreement | Pearson（诊断） |",
                "|---|---:|---:|---:|",
            ]
        )
    for feature_name, item in fitted_features.items():
        lines.append(
            f"| {feature_name} | {item['spearman']:.4f} | {item['pairwise_rank_agreement']:.4f} | {item['pearson']:.4f} |"
        )
    candidate_status = fit.get("candidate_status", {})
    if candidate_status:
        lines.extend(
            [
                "",
                "### 时间与有效性预筛",
                "",
                "| 候选 | 已评模型 | 主模型 API 时间(s) | 主模型 <420s | 软 guardrail 完整 | 本地提交有效 |",
                "|---|---:|---:|---|---|---|",
            ]
        )
        for candidate, item in candidate_status.items():
            lines.append(
                f"| {candidate} | {item['evaluated_models']}/{item['expected_models']} | {item['primary_api_total_seconds']:.3f} | {item['primary_panel_valid']} | {item['complete']} | {item['valid_submission']} |"
            )
    local_order = fit.get("local_primary_panel_order", fit.get("local_panel_order", []))
    if local_order:
        lines.extend(
            [
                "",
                "### Qwen 主模型 shaped-panel 排序",
                "",
                " > ".join(
                    f"{item['candidate']} ({item['score']:.6f})"
                    for item in local_order
                ),
            ]
        )
    native_order = fit.get("local_official_flow_order", [])
    if native_order:
        lines.extend(
            [
                "",
                "### Native 原始分（仅诊断）",
                "",
                " > ".join(
                    f"{item['candidate']} ({item['score']:.6f})"
                    for item in native_order
                ),
            ]
        )
    ordering = fit.get("c39_vs_c40", {})
    if ordering:
        lines.extend(["", "### C39 / C40 排序", ""])
    for model_name, item in ordering.items():
        lines.append(
            f"- `{model_name}`：C39 official-flow total={item['c39_official_flow_total']:.6f}，C40={item['c40_official_flow_total']:.6f}，C39>C40：`{item['c39_above_c40']}`。"
        )
    lines.extend(
        [
            "",
            "## 解释与使用边界",
            "",
            "1. 默认候选晋级看 Qwen 主模型的 `primary_panel_score_total`；Linear/Attention 目标权重固定为 250/200。其他模型的 panel 均值只作软 guardrail 和回归诊断。",
            "2. `official_flow_total` 仍完整保留，便于和旧报告逐位对比，但不再因模型层数或本地窗口数量差异直接主导排序。",
            "3. 本地数据不是官方隐藏数据，因此 shaped panel 只能用于相对排序；官方锚点只用于事后审计排序一致率，不能把 panel 分数线性换算成 Official Score。",
            "4. `synthetic_attention_eval.py` 不由本套件调用；它只能做接口/性质测试，不能用于候选排名。",
            "5. `cache_mode=read` 时本次结果只来自已保存的模型前向快照，不加载 tokenizer/model，也不读取网络；`cache_mode=write` 才会刷新快照。",
            "6. 本地时间按每个模型代理的六个正式 API 调用累计；主模型必须严格小于 420 秒，多模型代理时间不相加。",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Any]:
    started_at = time.strftime("%Y-%m-%d %H:%M:%S")
    selected_models = args.models or list(MODEL_SPECS)
    primary_model = args.primary_model
    primary_model_selection_warning = None
    if primary_model not in selected_models:
        # Keep single-model diagnostics convenient while making the fallback
        # explicit in JSON/report output.  The default full panel always
        # contains Qwen, so this does not silently replace the normal primary.
        primary_model_selection_warning = (
            f"requested primary model {primary_model!r} is not in --models; "
            f"using {selected_models[0]!r} as the local primary"
        )
        primary_model = selected_models[0]
    candidate_specs = dict(CANDIDATE_SPECS)
    if args.solution is not None:
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", args.candidate_name):
            raise ValueError("--candidate-name may contain only letters, digits, '.', '_' and '-'")
        if args.candidate_name in candidate_specs:
            raise ValueError(
                f"--candidate-name {args.candidate_name!r} conflicts with an official anchor"
            )
        candidate_specs[args.candidate_name] = CandidateSpec(
            args.candidate_name,
            args.solution.resolve(),
            None,
            None,
        )
        selected_candidates = list(args.candidates or []) + [args.candidate_name]
    else:
        selected_candidates = args.candidates or list(CANDIDATE_SPECS)
    unknown_models = [name for name in selected_models if name not in MODEL_SPECS]
    unknown_candidates = [name for name in selected_candidates if name not in candidate_specs]
    if unknown_models:
        raise ValueError(f"unknown models: {unknown_models}")
    if unknown_candidates:
        raise ValueError(f"unknown candidates: {unknown_candidates}")
    if args.layers is not None and args.layers <= 0:
        raise ValueError("--layers must be positive")
    if args.capture_only and args.cache_mode == "off":
        raise ValueError("--capture-only requires --cache-mode auto, read, or write")

    run_metadata: dict[str, Any] = {
        "started_at": started_at,
        "mode": args.mode,
        "sequence_length": args.seq,
        "calibration_samples": args.calib,
        "test_samples": args.test,
        "device": args.device,
        "algorithm_device": args.algorithm_device or args.device,
        "data_dir": str(args.data_dir),
        "cache_dir": str(args.cache_dir),
        "cache_mode": args.cache_mode,
        "capture_only": args.capture_only,
        "layers": args.layers,
        "models": selected_models,
        "panel_profile": args.panel_profile,
        "primary_model": primary_model,
        "primary_model_requested": args.primary_model,
        "primary_model_selection_warning": primary_model_selection_warning,
        "reference_panel": {
            "revision": OFFICIAL_PANEL_REVISION,
            "linear_cases": REFERENCE_PANEL_LINEAR_CASES,
            "attention_cases": REFERENCE_PANEL_ATTENTION_CASES,
            "total_cases": REFERENCE_PANEL_TOTAL_CASES,
            "case_projection": (
                "component_mean_preserving; no local case duplication or official "
                "absolute-score conversion"
            ),
        },
        "official_reference_context": {
            "revised_anchors": OFFICIAL_PANEL_REVISION,
            "external": list(EXTERNAL_OFFICIAL_REFERENCES),
            "extra_cases": dict(OFFICIAL_EXTRA_CASE_REFERENCE),
        },
        "candidates": selected_candidates,
        "candidate_sources": {
            name: str(candidate_specs[name].path) for name in selected_candidates
        },
        "official_runtime_limit_seconds": OFFICIAL_RUNTIME_LIMIT_SECONDS,
        "scoring_protocol": {
            "version": SCORING_PROTOCOL_VERSION,
            "case_formula": "(MSE_STD-MSE_PLAYER)/MSE_STD",
            "aggregation": (
                "qwen-primary fixed panel: 250*Linear_mean + "
                "200*Attention_mean"
                if args.panel_profile == "qwen-official"
                else "sum_all_native_linear_and_attention_cases"
            ),
            "native_aggregation": "sum_all_linear_and_attention_cases",
            "attention_causal": False,
            "candidate_private_helpers_used_for_scoring": False,
            "standard_codec_source": str(STANDARD_CODEC_PATH),
            "standard_codec_sha256": sha256_file(STANDARD_CODEC_PATH),
        },
    }
    model_status: list[dict[str, Any]] = []
    results: list[dict[str, Any]] = []
    partial_path = args.output.with_suffix(".partial.json")

    def persist_partial() -> None:
        partial_fit = audit_official_ranking(
            results,
            selected_candidates,
            candidate_specs,
            selected_models,
            args.panel_profile,
            primary_model,
        )
        _write_json(
            partial_path,
            {
                "run": run_metadata,
                "model_status": model_status,
                "results": results,
                "official_ranking_audit": partial_fit,
                "partial": True,
            },
        )

    for model_name in selected_models:
        spec = MODEL_SPECS[model_name]
        try:
            data, data_source, cache_path = load_or_collect_model_data(
                spec,
                args.data_dir,
                args.cache_dir,
                args.seq,
                args.calib,
                args.test,
                args.device,
                args.layers,
                args.cache_mode,
            )
            model_status.append(
                {
                    "model": model_name,
                    "status": "loaded",
                    "source": data_source,
                    "cache_path": str(cache_path),
                    "metadata": data.metadata,
                }
            )
            print(
                f"{model_name}: data_source={data_source} cache={cache_path}",
                flush=True,
            )
        except Exception as exc:
            status = {"model": model_name, "status": "skipped", "error": f"{type(exc).__name__}: {exc}"}
            model_status.append(status)
            persist_partial()
            if args.strict_models or args.cache_mode == "read":
                raise
            continue
        if args.capture_only:
            persist_partial()
            print(f"{model_name}: capture-only complete", flush=True)
            del data
            continue
        for candidate_name in selected_candidates:
            candidate = candidate_specs[candidate_name]
            try:
                result = evaluate_candidate(
                    candidate,
                    data,
                    args.mode,
                    args.algorithm_device or args.device,
                    args.panel_profile,
                )
                results.append(result)
                persist_partial()
                print(
                    f"{model_name:14s} {candidate_name:4s} "
                    f"official-total={result['official_flow_score']['total']:.6f} "
                    f"panel-total={result['panel_score']['total']:.6f} "
                    f"linear={result['official_flow_score']['linear']:.6f} "
                    f"attention={result['official_flow_score']['attention']:.6f} "
                    f"api={result['timing']['official_api_total_seconds']:.2f}s",
                    flush=True,
                )
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
                results.append(
                    {
                        "model": model_name,
                        "candidate": candidate_name,
                        "error": error,
                    }
                )
                persist_partial()
                print(
                    f"{model_name:14s} {candidate_name:4s} ERROR {error}",
                    flush=True,
                )
        del data

    valid_results = [item for item in results if "error" not in item]
    fit = audit_official_ranking(
        results,
        selected_candidates,
        candidate_specs,
        selected_models,
        args.panel_profile,
        primary_model,
    )
    run_metadata["finished_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    run_metadata["loaded_models"] = [item["model"] for item in model_status if item["status"] == "loaded"]
    run_metadata["result_count"] = len(valid_results)
    run_metadata["official_flow_valid"] = bool(fit.get("candidate_status")) and all(
        item["valid_submission"] for item in fit["candidate_status"].values()
    )
    run_metadata["panel_valid"] = run_metadata["official_flow_valid"]
    output = {
        "run": run_metadata,
        "model_status": model_status,
        "results": results,
        "official_ranking_audit": fit,
    }
    output_path = args.output
    _write_json(output_path, output)
    write_report(args.report, run_metadata, model_status, valid_results, fit)
    print(f"JSON: {output_path}")
    print(f"REPORT: {args.report}")
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--models",
        nargs="+",
        choices=tuple(MODEL_SPECS),
        default=None,
        help=(
            "models to evaluate (default: every manifest model that is available "
            "locally); Qwen is the primary panel model"
        ),
    )
    parser.add_argument(
        "--panel-profile",
        choices=PANEL_PROFILES,
        default=DEFAULT_PANEL_PROFILE,
        help=(
            "local aggregation profile; qwen-official preserves component means "
            "at 250 Linear/200 Attention, native keeps raw case sums"
        ),
    )
    parser.add_argument(
        "--primary-model",
        choices=tuple(MODEL_SPECS),
        default=DEFAULT_PRIMARY_MODEL,
        help=(
            "model driving the shaped-panel ranking (default: Qwen2.5-0.5B); "
            "when omitted from --models, the first selected model is used"
        ),
    )
    parser.add_argument(
        "--candidates",
        nargs="+",
        choices=tuple(CANDIDATE_SPECS),
        default=None,
        help="registered anchors; may be combined with --solution for paired ranking",
    )
    parser.add_argument(
        "--solution",
        type=Path,
        default=None,
        help="evaluate an arbitrary solution.py, optionally beside --candidates anchors",
    )
    parser.add_argument(
        "--candidate-name",
        default="active",
        help="label used with --solution (default: active)",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=ROOT / "data" / WIKITEXT_CONFIG,
    )
    parser.add_argument("--seq", type=int, default=128)
    parser.add_argument("--calib", type=int, default=2)
    parser.add_argument("--test", type=int, default=4)
    parser.add_argument("--layers", type=int, default=None, help="optional layer cap for diagnostics")
    parser.add_argument("--mode", choices=("amax6", "amax4", "pow2"), default="amax6")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--algorithm-device",
        default=None,
        help="device for candidate calibration/scoring (default: same as --device)",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=DEFAULT_CACHE_DIR,
        help="directory for CPU model-forward snapshots",
    )
    parser.add_argument(
        "--cache-mode",
        choices=("auto", "read", "write", "off"),
        default="auto",
        help="auto=read or create; read=cache-only; write=refresh; off=do not persist",
    )
    parser.add_argument(
        "--capture-only",
        action="store_true",
        help="capture/validate model data and stop before invoking candidate algorithms",
    )
    parser.add_argument("--strict-models", action="store_true", help="fail instead of skipping a missing/broken model")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "artifacts" / "real_model_suite" / "latest.json",
    )
    parser.add_argument(
        "--report",
        type=Path,
        required=True,
        help=(
            "markdown report path; required so a run can never silently "
            "overwrite an archived evaluation report"
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if min(args.seq, args.calib, args.test) <= 0:
        raise SystemExit("--seq, --calib, and --test must be positive")
    output = run(args)
    if args.solution is not None and not output["run"]["official_flow_valid"]:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
