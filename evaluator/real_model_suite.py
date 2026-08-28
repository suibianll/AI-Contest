"""Evaluate compliant HiF4 candidates on several real language models.

This is a development evaluator, not an official-score replacement.  It has
three deliberate properties:

* calibration windows come from WikiText-2 train and test windows come from
  WikiText-2 validation; windows never wrap, repeat, or share a source
  document;
* every activation is captured from an actual local ``AutoModelForCausalLM``
  forward pass, with model-specific projection and rotary-embedding adapters;
* the evaluator computes the reference outputs only after a candidate has
  returned its quantization state.  The candidate receives weights and
  activations, never an evaluator output, residual, or fitted official score.

The candidate-side rule is therefore unchanged: a solution must not form
``A @ W`` to select or infer ``Q(A)``.  The output products in this file are
evaluator-side scoring references only.

Typical run (from the repository root)::

    python evaluator/real_model_suite.py --device cuda

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


WIKITEXT_REVISION = "b08601e04326c79dfdd32d625aee71d232d685c3"
WIKITEXT_CONFIG = "wikitext-2-raw-v1"
WIKITEXT_FILES = {
    "train": "train-00000-of-00001.parquet",
    "validation": "validation-00000-of-00001.parquet",
    "test": "test-00000-of-00001.parquet",
}


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
    official_score: int
    official_time: float


CANDIDATE_SPECS: dict[str, CandidateSpec] = {
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
        14613,
        159.2,
    ),
    "c40": CandidateSpec(
        "c40",
        ROOT / "solutions" / "20260828_v032_c40-robust-blockldlq_official-score14432_time216.667s" / "solution.py",
        14432,
        216.667,
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
            "data": data_metadata,
        }
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


def _linear_detail(
    solution: Any,
    weight_pair: tuple[torch.Tensor, torch.Tensor],
    activation_pairs: Sequence[tuple[torch.Tensor, torch.Tensor]],
    activation_state: Any,
    weight_params: Any,
) -> dict[str, float]:
    """Score a Linear case against the frozen evaluator-side reference."""

    weight_reference = solution._dequantize_nvfp4_float32(*weight_pair)
    weight_standard = std_hif4(solution, weight_reference)
    weight_player = solution._dequantize_hif4(weight_params).to(torch.float32)
    standard_sum = 0.0
    player_sum = 0.0
    relative: list[float] = []
    elements = 0
    for activation_pair in activation_pairs:
        activation_reference = solution._dequantize_nvfp4_float32(*activation_pair)
        reference = activation_reference @ weight_reference.T
        standard = std_hif4(solution, activation_reference) @ weight_standard.T
        player_activation = solution._dequantize_hif4(
            solution.hif4_dynamic_quantize_activation(
                *activation_pair, activation_state
            )
        ).to(torch.float32)
        player = player_activation @ weight_player.T
        standard_mse = float((standard - reference).square().mean())
        player_mse = float((player - reference).square().mean())
        standard_sum += standard_mse * reference.numel()
        player_sum += player_mse * reference.numel()
        elements += int(reference.numel())
        relative.append((standard_mse - player_mse) / max(standard_mse, 1.0e-30))
    return {
        "gain": sum(relative) / len(relative),
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
) -> dict[str, float]:
    relative: list[float] = []
    standard_sum = 0.0
    player_sum = 0.0
    elements = 0
    for q_pair, k_pair, v_pair in qkv_pairs:
        q_reference = solution._dequantize_nvfp4_float32(*q_pair)
        k_reference = solution._dequantize_nvfp4_float32(*k_pair)
        v_reference = solution._dequantize_nvfp4_float32(*v_pair)
        q_standard = std_hif4(solution, q_reference)
        k_standard = std_hif4(solution, k_reference)
        v_standard = std_hif4(solution, v_reference)
        q_player = solution._dequantize_hif4(
            solution.hif4_dynamic_quantize_q(*q_pair, q_heads, head_dim, q_state)
        ).to(torch.float32)
        k_player = solution._dequantize_hif4(
            solution.hif4_dynamic_quantize_k(*k_pair, kv_heads, head_dim, k_state)
        ).to(torch.float32)
        v_player = solution._dequantize_hif4(
            solution.hif4_dynamic_quantize_v(*v_pair, kv_heads, head_dim, v_state)
        ).to(torch.float32)
        reference = causal_attention(
            q_reference[None], k_reference[None], v_reference[None],
            q_heads, kv_heads, head_dim, True
        )
        standard = causal_attention(
            q_standard[None], k_standard[None], v_standard[None],
            q_heads, kv_heads, head_dim, True
        )
        player = causal_attention(
            q_player[None], k_player[None], v_player[None],
            q_heads, kv_heads, head_dim, True
        )
        standard_mse = float((standard - reference).square().mean())
        player_mse = float((player - reference).square().mean())
        standard_sum += standard_mse * reference.numel()
        player_sum += player_mse * reference.numel()
        elements += int(reference.numel())
        relative.append((standard_mse - player_mse) / max(standard_mse, 1.0e-30))
    return {
        "gain": sum(relative) / len(relative),
        "standard_sum": standard_sum,
        "player_sum": player_sum,
        "elements": elements,
    }


def _aggregate_details(details: Sequence[dict[str, float]]) -> dict[str, float]:
    if not details:
        return {"macro_gain": float("nan"), "global_gain": float("nan"), "cases": 0}
    standard = sum(item["standard_sum"] for item in details)
    player = sum(item["player_sum"] for item in details)
    return {
        "macro_gain": sum(item["gain"] for item in details) / len(details),
        "global_gain": (standard - player) / max(standard, 1.0e-30),
        "standard_sum": standard,
        "player_sum": player,
        "elements": sum(int(item["elements"]) for item in details),
        "cases": len(details),
    }


def evaluate_candidate(
    candidate: CandidateSpec,
    data: ModelData,
    mode: str,
    algorithm_device_name: str,
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
        "attention_causal": attention,
        "timing": {
            "wall_seconds": elapsed,
            "algorithm_stage_seconds": (
                None
                if stats["first_start"] is None
                else stats["last_end"] - stats["first_start"]
            ),
            "calibration_seconds": stats["calibration"],
            "dynamic_seconds": stats["dynamic"],
            "api_calls": stats["calls"],
            "nested_api_calls": stats["nested_calls"],
            "under_300_seconds": (
                stats["first_start"] is None
                or stats["last_end"] - stats["first_start"] < 300.0
            ),
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


def _ols(xs: Sequence[float], ys: Sequence[float]) -> dict[str, float]:
    if len(xs) < 2:
        return {"slope": float("nan"), "intercept": float("nan"), "r2": float("nan")}
    x_mean = _mean(xs)
    y_mean = _mean(ys)
    denom = sum((x - x_mean) ** 2 for x in xs)
    slope = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys)) / denom if denom else 0.0
    intercept = y_mean - slope * x_mean
    residual = sum((y - (intercept + slope * x)) ** 2 for x, y in zip(xs, ys))
    total = sum((y - y_mean) ** 2 for y in ys)
    return {
        "slope": slope,
        "intercept": intercept,
        "r2": 1.0 - residual / total if total else float("nan"),
    }


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


def fit_official_anchors(
    results: Sequence[dict[str, Any]],
    requested_candidates: Sequence[str],
) -> dict[str, Any]:
    by_model: dict[str, dict[str, dict[str, Any]]] = {}
    for result in results:
        by_model.setdefault(result["model"], {})[result["candidate"]] = result
    official = {
        name: CANDIDATE_SPECS[name].official_score
        for name in requested_candidates
        if name in CANDIDATE_SPECS
    }
    model_features: dict[str, dict[str, dict[str, float]]] = {}
    for model_name, candidate_results in by_model.items():
        model_features[model_name] = {}
        for feature_name, getter in (
            ("linear_global_gain", lambda item: item["linear"]["global_gain"]),
            ("linear_macro_gain", lambda item: item["linear"]["macro_gain"]),
            ("component_macro_gain", lambda item: item["linear_component_macro_gain"]),
            ("attention_causal_global_gain", lambda item: item["attention_causal"]["global_gain"]),
            ("attention_causal_macro_gain", lambda item: item["attention_causal"]["macro_gain"]),
        ):
            model_features[model_name][feature_name] = {
                candidate: float(getter(candidate_results[candidate]))
                for candidate in requested_candidates
                if candidate in candidate_results
            }

    aggregate_features: dict[str, dict[str, float]] = {}
    for feature_name in next(iter(model_features.values()), {}):
        aggregate_features[feature_name] = {
            candidate: _mean(
                model_features[model_name][feature_name][candidate]
                for model_name in model_features
                if candidate in model_features[model_name].get(feature_name, {})
            )
            for candidate in requested_candidates
        }

    fit: dict[str, Any] = {}
    for feature_name, values in aggregate_features.items():
        names = [candidate for candidate in requested_candidates if candidate in official and math.isfinite(values[candidate])]
        xs = [values[name] for name in names]
        ys = [float(official[name]) for name in names]
        ols = _ols(xs, ys)
        loo_errors: list[float] = []
        if len(names) >= 3:
            for held_out in range(len(names)):
                train_x = xs[:held_out] + xs[held_out + 1 :]
                train_y = ys[:held_out] + ys[held_out + 1 :]
                params = _ols(train_x, train_y)
                prediction = params["intercept"] + params["slope"] * xs[held_out]
                loo_errors.append(abs(prediction - ys[held_out]))
        fit[feature_name] = {
            "candidates": names,
            "local_values": {name: values[name] for name in names},
            "official_scores": {name: official[name] for name in names},
            "pearson": pearson(xs, ys),
            "spearman": spearman(xs, ys),
            "pairwise_rank_agreement": _pairwise_rank_agreement(xs, ys),
            "ols": ols,
            "leave_one_out_mae": _mean(loo_errors) if loo_errors else float("nan"),
        }

    ordering = {}
    for model_name, values in model_features.items():
        c39 = values.get("linear_global_gain", {}).get("c39", float("nan"))
        c40 = values.get("linear_global_gain", {}).get("c40", float("nan"))
        ordering[model_name] = {
            "c39_linear_global_gain": c39,
            "c40_linear_global_gain": c40,
            "c39_above_c40": bool(math.isfinite(c39) and math.isfinite(c40) and c39 > c40),
        }
    aggregate_order = aggregate_features.get("linear_global_gain", {})
    ordering["aggregate"] = {
        "c39_linear_global_gain": aggregate_order.get("c39", float("nan")),
        "c40_linear_global_gain": aggregate_order.get("c40", float("nan")),
        "c39_above_c40": bool(
            math.isfinite(aggregate_order.get("c39", float("nan")))
            and math.isfinite(aggregate_order.get("c40", float("nan")))
            and aggregate_order["c39"] > aggregate_order["c40"]
        ),
    }
    return {
        "official_anchor_scores": official,
        "model_features": model_features,
        "aggregate_features": aggregate_features,
        "fit": fit,
        "c39_vs_c40": ordering,
        "warning": "four official anchors are a diagnostic, not a validated score mapping",
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
    lines = [
        "# 多模型真实语料评估校准报告",
        "",
        f"运行时间：{run_metadata['started_at']}（配置 mode={run_metadata['mode']}，seq={run_metadata['sequence_length']}，calib={run_metadata['calibration_samples']}，test={run_metadata['test_samples']}）",
        "",
        "本报告只用于检查本地评估器是否能复现已有官方候选的相对方向。官方分数没有进入候选校准状态，也没有传给 `solution.py`。评估器内部的输出矩阵乘法只在候选返回量化结果之后，用作固定参考误差。",
        "",
        "## 数据与模型完整性",
        "",
        f"- 数据集：`Salesforce/wikitext` / `{WIKITEXT_CONFIG}` / revision `{WIKITEXT_REVISION}`。",
        "- calibration 来自 train，test 来自 validation；每个窗口来自一个文档，禁止环形重复、窗口重叠和跨 split 文档复用。",
        "- 模型状态：",
        "",
        "| 模型 | 状态 | 层数 | hidden | heads / kv-heads | 说明 |",
        "|---|---|---:|---:|---:|---|",
    ]
    for status in model_status:
        if status.get("status") == "loaded":
            metadata = status["metadata"]
            lines.append(
                f"| {status['model']} | loaded | {metadata['layers']} | {metadata['hidden_size']} | {metadata['q_heads']} / {metadata['kv_heads']} | {metadata['family']} |"
            )
        else:
            lines.append(
                f"| {status['model']} | skipped | - | - | - | {status.get('error', 'unknown error')} |"
            )

    lines.extend(
        [
            "",
            "## 候选在各模型上的结果",
            "",
            "`linear-global` 按 evaluator reference MSE 的元素数加权；`component-macro` 先按 q/k/v/o/fc/proj 聚合再平均，避免 Qwen 的 gate/up 两个投影重复放大 FFN；`attention-causal` 使用真实模型的 Q/K/V（含模型自身 RoPE/GQA 适配）。",
            "",
            "| 模型 | 候选 | linear-global | component-macro | attention-causal | algorithm-stage(s) | <300s |",
            "|---|---|---:|---:|---:|---:|---|",
        ]
    )
    for result in results:
        timing = result["timing"]
        lines.append(
            f"| {result['model']} | {result['candidate']} | {result['linear']['global_gain']:.6f} | {result['linear_component_macro_gain']:.6f} | {result['attention_causal']['global_gain']:.6f} | {timing['algorithm_stage_seconds']:.3f} | {timing['under_300_seconds']} |"
        )

    lines.extend(
        [
            "",
            "## 与官方锚点的拟合诊断",
            "",
            "官方锚点：C21=14437、C38=14092、C39=14613、C40=14432。下表先对已加载模型取均值，再与四个官方分数计算相关性；样本只有四个，不能据此拟合可靠的绝对分数换算公式。",
            "",
            "| 本地特征 | Pearson | Spearman | pairwise rank agreement | OLS R² | leave-one-out MAE |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for feature_name, item in fit.get("fit", {}).items():
        ols = item["ols"]
        lines.append(
            f"| {feature_name} | {item['pearson']:.4f} | {item['spearman']:.4f} | {item['pairwise_rank_agreement']:.4f} | {ols['r2']:.4f} | {item['leave_one_out_mae']:.2f} |"
        )
    lines.extend(["", "### C39 / C40 排序", ""])
    for model_name, item in fit.get("c39_vs_c40", {}).items():
        lines.append(
            f"- `{model_name}`：C39 linear-global={item['c39_linear_global_gain']:.6f}，C40={item['c40_linear_global_gain']:.6f}，C39>C40：`{item['c39_above_c40']}`。"
        )
    lines.extend(
        [
            "",
            "## 解释与使用边界",
            "",
            "1. 只有当多个模型、多个特征同时保持方向，并且至少复现 C39 高于 C40 的已知官方排序时，才把本地分数当作候选筛选信号。",
            "2. 如果某个特征只在 GPT-2-small 上有效，或 C38/C40 的排序反转，应优先检查数据分割、架构适配和聚合口径，不应继续调候选阈值。",
            "3. `synthetic_attention_eval.py` 不由本套件调用；它只能做接口/性质测试，不能用于候选排名。",
            "4. algorithm-stage 是候选 API 的开发计时，官方端到端计时仍以赛事评测为准；报告中的 `<300s` 只是本地硬约束预筛。",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Any]:
    started_at = time.strftime("%Y-%m-%d %H:%M:%S")
    selected_models = args.models or list(MODEL_SPECS)
    selected_candidates = args.candidates or list(CANDIDATE_SPECS)
    unknown_models = [name for name in selected_models if name not in MODEL_SPECS]
    unknown_candidates = [name for name in selected_candidates if name not in CANDIDATE_SPECS]
    if unknown_models:
        raise ValueError(f"unknown models: {unknown_models}")
    if unknown_candidates:
        raise ValueError(f"unknown candidates: {unknown_candidates}")
    if args.layers is not None and args.layers <= 0:
        raise ValueError("--layers must be positive")

    run_metadata: dict[str, Any] = {
        "started_at": started_at,
        "mode": args.mode,
        "sequence_length": args.seq,
        "calibration_samples": args.calib,
        "test_samples": args.test,
        "device": args.device,
        "algorithm_device": args.algorithm_device or args.device,
        "data_dir": str(args.data_dir),
        "models": selected_models,
        "candidates": selected_candidates,
        "official_runtime_limit_seconds": 300.0,
    }
    model_status: list[dict[str, Any]] = []
    results: list[dict[str, Any]] = []
    partial_path = args.output.with_suffix(".partial.json")

    def persist_partial() -> None:
        partial_fit = fit_official_anchors(
            [item for item in results if "error" not in item], selected_candidates
        )
        _write_json(
            partial_path,
            {
                "run": run_metadata,
                "model_status": model_status,
                "results": results,
                "official_fit": partial_fit,
                "partial": True,
            },
        )

    for model_name in selected_models:
        spec = MODEL_SPECS[model_name]
        try:
            data = collect_model_data(
                spec,
                args.data_dir,
                args.seq,
                args.calib,
                args.test,
                args.device,
                args.layers,
            )
            model_status.append({"model": model_name, "status": "loaded", "metadata": data.metadata})
        except Exception as exc:
            status = {"model": model_name, "status": "skipped", "error": f"{type(exc).__name__}: {exc}"}
            model_status.append(status)
            persist_partial()
            if args.strict_models:
                raise
            continue
        for candidate_name in selected_candidates:
            candidate = CANDIDATE_SPECS[candidate_name]
            try:
                result = evaluate_candidate(
                    candidate, data, args.mode, args.algorithm_device or args.device
                )
                results.append(result)
                persist_partial()
                print(
                    f"{model_name:14s} {candidate_name:4s} "
                    f"linear={result['linear']['global_gain']:.6f} "
                    f"component={result['linear_component_macro_gain']:.6f} "
                    f"attention={result['attention_causal']['global_gain']:.6f} "
                    f"stage={result['timing']['algorithm_stage_seconds']:.2f}s",
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
    fit = fit_official_anchors(valid_results, selected_candidates)
    run_metadata["finished_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    run_metadata["loaded_models"] = [item["model"] for item in model_status if item["status"] == "loaded"]
    run_metadata["result_count"] = len(valid_results)
    output = {
        "run": run_metadata,
        "model_status": model_status,
        "results": results,
        "official_fit": fit,
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
        help="models to evaluate (default: every manifest model that is available locally)",
    )
    parser.add_argument(
        "--candidates",
        nargs="+",
        choices=tuple(CANDIDATE_SPECS),
        default=None,
        help="official-anchor candidates (default: c21 c38 c39 c40)",
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
    parser.add_argument("--strict-models", action="store_true", help="fail instead of skipping a missing/broken model")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "artifacts" / "real_model_suite" / "latest.json",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=ROOT / "docs" / "real-model-evaluator-calibration-2026-08-28.md",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if min(args.seq, args.calib, args.test) <= 0:
        raise SystemExit("--seq, --calib, and --test must be positive")
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
