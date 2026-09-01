"""Cross-model HiF4 probe for local GPT-family checkpoints.

This is intentionally separate from :mod:`official_eval`'s default Qwen
``proxy-v2`` panel.  The competition brief specifies tensors and API calls,
not a public model architecture.  A local GPT-2 run therefore answers a
different question: does the same candidate remain well behaved when the
layer shapes, FFN topology, head count and positional encoding change?

The adapter exposes GPT-2's native operations without fabricating a gated
FFN role:

* ``attn.c_attn`` is split into independent Q/K/V matrices;
* ``attn.c_proj`` is the attention output projection;
* ``mlp.c_fc`` is represented once as ``ffn_in`` (GPT-2 GELU has no gate);
* ``mlp.c_proj`` is represented as ``proj``.

Scores are still computed with the evaluator's real NVFP4/HiF4 tensors and
the candidate's six APIs.  They are *cross-model proxy diagnostics*, never an
official score or a replacement for the Qwen panel.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Sequence

import torch

import official_eval as evaluator  # noqa: E402
from official_eval import CALIBRATION_LENGTHS, DATA_DIR, PROTOCOL, ROOT, TEST_LENGTHS, WIKITEXT_FILES, RawPack  # noqa: E402


GPT_ROLES = ("q", "k", "v", "o", "ffn_in", "proj")
PROBE_PROTOCOL = "cross-model-probe-v1"
SUPPORTED_MODELS = ("gpt2", "gpt2-medium")


def _cpu_float(value: torch.Tensor) -> torch.Tensor:
    return value.detach().to(device="cpu", dtype=torch.float32).contiguous()


def _flat(value: torch.Tensor) -> torch.Tensor:
    return _cpu_float(value).reshape(-1, value.shape[-1]).contiguous()


def _conv1d_native(module: torch.nn.Module) -> torch.Tensor:
    """Transformers' Conv1D stores [in, out], unlike the public [out, in]."""

    return _cpu_float(module.weight).transpose(0, 1).contiguous()


def _load_gpt2(model_name: str, device: torch.device) -> tuple[Any, torch.nn.Module, Path]:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_path = ROOT / "models" / model_name
    if not model_path.is_dir():
        raise FileNotFoundError(f"model directory does not exist: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, use_fast=True)
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        local_files_only=True,
        dtype=dtype,
    )
    model.eval().to(device)
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False
    if getattr(model.config, "model_type", None) != "gpt2":
        raise ValueError(f"{model_name} is not a GPT-2 checkpoint")
    return tokenizer, model, model_path


def _capture_gpt2_windows(
    model: torch.nn.Module,
    windows: Sequence[evaluator.Window],
    device: torch.device,
) -> tuple[dict[str, list[list[torch.Tensor]]], list[list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]]]:
    """Capture the same API operands as Qwen, using GPT-2's fused attention."""

    blocks = list(model.transformer.h)
    hidden = int(model.config.n_embd)
    captures: list[dict[str, torch.Tensor]] = [{} for _ in blocks]
    handles: list[Any] = []

    def capture_module(name: str, module: torch.nn.Module, layer_index: int, output: bool) -> None:
        def hook(
            _module: torch.nn.Module,
            inputs: tuple[Any, ...],
            value: Any,
            index: int = layer_index,
        ) -> None:
            if not inputs or not torch.is_tensor(inputs[0]):
                raise RuntimeError(f"GPT-2 {name} received no tensor input")
            captures[index][f"{name}_in"] = _cpu_float(inputs[0])
            if output:
                if isinstance(value, (tuple, list)):
                    value = value[0]
                if not torch.is_tensor(value):
                    raise RuntimeError(f"GPT-2 {name} returned no tensor output")
                captures[index][f"{name}_out"] = _cpu_float(value)

        handles.append(module.register_forward_hook(hook))

    for index, block in enumerate(blocks):
        capture_module("attn_c_attn", block.attn.c_attn, index, output=True)
        capture_module("attn_c_proj", block.attn.c_proj, index, output=False)
        capture_module("mlp_c_fc", block.mlp.c_fc, index, output=False)
        capture_module("mlp_c_proj", block.mlp.c_proj, index, output=False)

    activations = {role: [] for role in GPT_ROLES}
    qkv_store: list[list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]] = []
    try:
        for window in windows:
            captures = [{} for _ in blocks]
            input_ids = torch.tensor(window.input_ids, dtype=torch.long, device=device).unsqueeze(0)
            with torch.no_grad():
                model(input_ids=input_ids, use_cache=False)
            per_layer_qkv: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
            per_layer_activations = {role: [] for role in GPT_ROLES}
            for index, captured in enumerate(captures):
                required = {
                    "attn_c_attn_in",
                    "attn_c_attn_out",
                    "attn_c_proj_in",
                    "mlp_c_fc_in",
                    "mlp_c_proj_in",
                }
                missing = required - set(captured)
                if missing:
                    raise RuntimeError(f"GPT-2 layer {index} capture missing {sorted(missing)}")
                fused = _flat(captured["attn_c_attn_out"])
                if int(fused.shape[-1]) != 3 * hidden:
                    raise RuntimeError(
                        f"GPT-2 layer {index} fused c_attn width {int(fused.shape[-1])} != {3 * hidden}"
                    )
                q, k, v = fused.split(hidden, dim=-1)
                # GPT-2 uses absolute position embeddings and has no RoPE.
                per_layer_qkv.append((q.contiguous(), k.contiguous(), v.contiguous()))
                per_layer_activations["q"].append(_flat(captured["attn_c_attn_in"]))
                per_layer_activations["k"].append(_flat(captured["attn_c_attn_in"]))
                per_layer_activations["v"].append(_flat(captured["attn_c_attn_in"]))
                per_layer_activations["o"].append(_flat(captured["attn_c_proj_in"]))
                per_layer_activations["ffn_in"].append(_flat(captured["mlp_c_fc_in"]))
                per_layer_activations["proj"].append(_flat(captured["mlp_c_proj_in"]))
            for role in GPT_ROLES:
                activations[role].append(per_layer_activations[role])
            qkv_store.append(per_layer_qkv)
    finally:
        for handle in handles:
            handle.remove()
    return activations, qkv_store


def capture_gpt2_pack(model_name: str, device_name: str) -> RawPack:
    device = torch.device(device_name)
    tokenizer, model, model_path = _load_gpt2(model_name, device)
    try:
        paths = {split: DATA_DIR / filename for split, filename in WIKITEXT_FILES.items()}
        missing = [str(path) for path in paths.values() if not path.is_file()]
        if missing:
            raise FileNotFoundError("missing pinned WikiText files: " + ", ".join(missing))
        train_rows = evaluator._load_rows(paths["train"])
        validation_rows = evaluator._load_rows(paths["validation"])
        test_rows = evaluator._load_rows(paths["test"])
        calibration_windows = evaluator._select_calibration_windows(tokenizer, train_rows)
        test_windows = evaluator._select_test_windows(
            tokenizer,
            {"validation": validation_rows, "test": test_rows},
        )
        blocks = list(model.transformer.h)
        weights: list[dict[str, torch.Tensor]] = []
        for block in blocks:
            fused = _conv1d_native(block.attn.c_attn)
            if tuple(fused.shape) != (3 * int(model.config.n_embd), int(model.config.n_embd)):
                raise RuntimeError(f"unexpected GPT-2 c_attn shape {tuple(fused.shape)}")
            hidden = int(model.config.n_embd)
            weights.append({
                "q": fused[:hidden],
                "k": fused[hidden:2 * hidden],
                "v": fused[2 * hidden:],
                "o": _conv1d_native(block.attn.c_proj),
                "ffn_in": _conv1d_native(block.mlp.c_fc),
                "proj": _conv1d_native(block.mlp.c_proj),
            })
        calibration_activations, calibration_qkv = _capture_gpt2_windows(
            model, calibration_windows, device
        )
        test_activations, test_qkv = _capture_gpt2_windows(model, test_windows, device)
        q_heads = int(model.config.n_head)
        head_dim = hidden // q_heads
        metadata = {
            "protocol": PROBE_PROTOCOL,
            "base_protocol": PROTOCOL,
            "model": model_name,
            "model_path": str(model_path),
            "model_revision": "local-checkpoint",
            "architecture": "GPT2LMHeadModel",
            "dataset": "Salesforce/wikitext",
            "dataset_config": evaluator.WIKITEXT_CONFIG,
            "dataset_revision": evaluator.WIKITEXT_REVISION,
            "calibration_lengths": list(CALIBRATION_LENGTHS),
            "test_lengths": list(TEST_LENGTHS),
            "test_window_count": len(test_windows),
            "test_splits": sorted({window.split for window in test_windows}),
            "capture_device": str(device),
            "weights_dtype": "float32",
            "weight_layout": "[out_features, in_features]",
            "input_codec": evaluator.NVFP4_INPUT_CODEC,
            "input_mode": evaluator.NVFP4_MODE,
            "linear_roles": list(GPT_ROLES),
            "role_mapping": {
                "q/k/v": "fused attn.c_attn split into three independent matrices",
                "o": "attn.c_proj",
                "ffn_in": "mlp.c_fc (single GELU projection; no gate/up duplicate)",
                "proj": "mlp.c_proj",
            },
            "positional_encoding": "absolute learned position embeddings; no RoPE transform",
            "attention_formula": "public full softmax QK^T V; GPT-2 causal mask is not applied in the proxy score",
            "attention_heads": {"q": q_heads, "kv": q_heads, "head_dim": head_dim},
            "data_sha256": {split: evaluator.sha256_file(path) for split, path in paths.items()},
        }
        return RawPack(
            weights,
            calibration_activations,
            test_activations,
            calibration_qkv,
            test_qkv,
            calibration_windows,
            test_windows,
            len(blocks),
            hidden,
            q_heads,
            q_heads,
            head_dim,
            metadata,
            GPT_ROLES,
        )
    finally:
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def _write_report(path: Path, payload: dict[str, Any]) -> None:
    score = payload.get("score", {})
    timing = payload.get("timing", {})
    metadata = payload.get("data_metadata", {})
    decomposition = payload.get("decomposition", {})
    linear = decomposition.get("linear", {}).get("overall", {})
    attention = decomposition.get("attention", {}).get("overall", {})
    lines = [
        "# Cross-model GPT probe",
        "",
        f"- model: `{metadata.get('model')}` (`{metadata.get('architecture')}`)",
        f"- protocol: `{payload.get('protocol')}`; base codec: `{metadata.get('base_protocol')}`",
        f"- layers/hidden/heads: `{payload.get('layers')}/{payload.get('hidden_size')}/{payload.get('q_heads')}/{payload.get('head_dim')}` (q/kv)",
        f"- roles: `{metadata.get('linear_roles')}`",
        f"- panel: `{score.get('linear_cases', 0)} Linear + {score.get('attention_cases', 0)} Attention`",
        "- this is an architecture stress test, not an official score and not mixed into Qwen proxy trend audits",
        "",
        "| Linear mean | Attention mean | Overall mean | API total (s) | Wall (s) |",
        "|---:|---:|---:|---:|---:|",
        f"| {score.get('linear_mean', 0.0):.6f} | {score.get('attention_mean', 0.0):.6f} | {score.get('overall_mean', 0.0):.6f} | {timing.get('api_total_seconds', 0.0):.3f} | {timing.get('wall_seconds', 0.0):.3f} |",
        "",
        "## Error-source decomposition",
        "",
        f"- Linear interpretation: `{linear.get('interpretation', 'unknown')}`",
        f"- Linear W-only/A-only/Both/interaction: `{linear.get('gain', {}).get('w_only', 0.0):.6f}` / `{linear.get('gain', {}).get('a_only', 0.0):.6f}` / `{linear.get('gain', {}).get('both', 0.0):.6f}` / `{linear.get('gain', {}).get('interaction', 0.0):.6f}`",
        f"- Attention interpretation: `{attention.get('interpretation', 'unknown')}`",
        f"- Attention Q-only/K-only/V-only/QK-only/Both: `{attention.get('gain', {}).get('q_only', 0.0):.6f}` / `{attention.get('gain', {}).get('k_only', 0.0):.6f}` / `{attention.get('gain', {}).get('v_only', 0.0):.6f}` / `{attention.get('gain', {}).get('qk_only', 0.0):.6f}` / `{attention.get('gain', {}).get('both', 0.0):.6f}`",
        "",
        "Full per-role/layer/length results are in JSON `case_scores` and `decomposition`.",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Any]:
    model_name = str(args.model)
    if model_name not in SUPPORTED_MODELS:
        raise ValueError(f"unsupported model {model_name!r}; choose from {SUPPORTED_MODELS}")
    cache_path = args.cache.resolve()
    capture_start = time.perf_counter()
    if args.cache_mode == "read":
        raw = evaluator.load_pack(cache_path)
        if raw.metadata.get("model") != model_name:
            raise RuntimeError(
                f"cache model {raw.metadata.get('model')!r} does not match --model {model_name!r}"
            )
        source = "cache"
    else:
        raw = capture_gpt2_pack(model_name, args.capture_device)
        source = "model_forward"
        if args.cache_mode in {"auto", "write"}:
            evaluator.save_pack(raw, cache_path)
    prepared = evaluator.prepare_pack(
        raw,
        linear_count=args.linear_cases,
        attention_count=args.attention_cases,
        full_cases=args.full_cases,
    )
    capture_seconds = time.perf_counter() - capture_start
    source_path = (ROOT / args.solution).resolve()
    print(f"[cross-model:{model_name}] evaluating {source_path}", flush=True)
    result = evaluator.evaluate_solution(
        source_path,
        prepared,
        args.algorithm_device,
        decomposition=args.decomposition,
    )
    result["candidate"] = args.name
    result["official"] = {
        "status": "not_applicable",
        "score": None,
        "time_seconds": None,
        "note": "cross-model GPT probe; no official model/score claim",
    }
    payload = {
        "protocol": PROBE_PROTOCOL,
        "base_protocol": PROTOCOL,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "data_source": source,
        "capture_seconds": capture_seconds,
        "cache": str(cache_path),
        "model": model_name,
        "candidate": args.name,
        "layers": prepared.layers,
        "hidden_size": prepared.hidden_size,
        "q_heads": prepared.q_heads,
        "kv_heads": prepared.kv_heads,
        "head_dim": prepared.head_dim,
        "roles": list(prepared.roles),
        "data_metadata": prepared.metadata,
        "score": result["score"],
        "timing": result["timing"],
        "case_scores": result["case_scores"],
        "decomposition": result["decomposition"],
        "diagnostic_config": result["diagnostic_config"],
        "result": result,
    }
    _write_json(args.output, payload)
    if args.report:
        _write_report(args.report, payload)
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=SUPPORTED_MODELS, default="gpt2")
    parser.add_argument("--solution", type=Path, default=Path("solution.py"))
    parser.add_argument("--name", default="candidate")
    parser.add_argument("--cache", type=Path)
    parser.add_argument("--cache-mode", choices=("auto", "read", "write", "off"), default="auto")
    parser.add_argument("--linear-cases", type=int, default=None)
    parser.add_argument("--attention-cases", type=int, default=None)
    parser.add_argument("--full-cases", action="store_true")
    parser.add_argument("--capture-device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--algorithm-device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--no-decomposition", dest="decomposition", action="store_false")
    parser.set_defaults(decomposition=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--report", type=Path)
    return parser


def _defaults(args: argparse.Namespace) -> None:
    if args.cache is None:
        args.cache = ROOT / "artifacts" / "official_eval" / "cache" / f"{args.model}-{PROBE_PROTOCOL}.pt"
    if args.output is None:
        args.output = ROOT / "artifacts" / "official_eval" / f"{args.model}-cross-model.json"
    if args.report is None:
        args.report = ROOT / "logs" / "official_eval" / f"{args.model}-cross-model.md"


if __name__ == "__main__":
    parsed = build_parser().parse_args()
    _defaults(parsed)
    output = run(parsed)
    raise SystemExit(0 if output.get("result", {}).get("status", "ok") == "ok" else 1)
