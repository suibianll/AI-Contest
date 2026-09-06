"""J0 joint output-gauge oracle for the v186 HiF4 parent.

The probe is deliberately kept outside ``solution.py``.  It tests one fixed
paired diagonal gauge family in the final transformed coordinates and uses
the evaluator-owned legal codec for every encoded block.  Gauge choices are
selected on calibration folds only; the held-out window is used only for the
reported leave-one-window-out result.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch


ROOT = Path(__file__).resolve().parents[1]
for _path in (ROOT / "evaluator", ROOT, ROOT / "workbench"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import official_eval as v2  # noqa: E402
import proxy_v3_eval as v3  # noqa: E402
import reference_hif4 as ref  # noqa: E402
import solution as sol  # noqa: E402
from hierarchy_partition_activation_probe import _batched_exact_blocks  # noqa: E402
from legal_codec_output_probe import (  # noqa: E402
    _load_parent_attention_state,
    _load_parent_state,
    _state_activation_transform,
    _state_parent_transform,
    exact_legal_blocks,
    sha256_file,
)


EPS = 1.0e-12
GAUGE_EXPONENTS = (-0.25, 0.0, 0.25)
GAUGE_VALUES = tuple(2.0 ** exponent for exponent in GAUGE_EXPONENTS)


def _git_head() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True,
            capture_output=True, text=True,
        )
        return result.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _torch_environment(device: str) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "torch": str(torch.__version__),
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_version": str(torch.version.cuda),
        "requested_device": str(device),
    }
    if torch.cuda.is_available():
        index = torch.cuda.current_device()
        payload.update({
            "cuda_device_index": int(index),
            "gpu": torch.cuda.get_device_name(index),
            "gpu_capability": list(torch.cuda.get_device_capability(index)),
        })
    else:
        payload["gpu"] = None
    return payload


def _uniform_indices(total: int, count: int) -> tuple[int, ...]:
    if total <= 0 or count <= 0:
        return ()
    if total == 1 or count == 1:
        return (0,)
    return tuple(sorted({int(math.floor(k * (total - 1) / (count - 1))) for k in range(count)}))


def _loo_folds(window_count: int) -> tuple[tuple[int, tuple[int, ...]], ...]:
    if window_count < 2:
        raise ValueError("J0 requires at least two calibration windows")
    return tuple(
        (holdout, tuple(index for index in range(window_count) if index != holdout))
        for holdout in range(window_count)
    )


def _load_config(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("config must be an object")
    return value


def _manifest(
    args: argparse.Namespace,
    config_path: Path,
    cache_path: Path,
    parent_path: Path,
    expected: Mapping[str, Any],
    executed: Mapping[str, Any],
    status: str,
) -> dict[str, Any]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    tool_path = Path(__file__).resolve()
    reference_path = ROOT / "evaluator" / "reference_hif4.py"
    return {
        "stage": "j0",
        "status": status,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "git_head": _git_head(),
        "parent": str(parent_path.resolve()),
        "parent_sha256": sha256_file(parent_path),
        "tool": str(tool_path),
        "tool_sha256": sha256_file(tool_path),
        "config": str(config_path.resolve()),
        "config_sha256": sha256_file(config_path),
        "cache": str(cache_path.resolve()),
        "cache_sha256": sha256_file(cache_path),
        "reference_hif4": str(reference_path.resolve()),
        "reference_hif4_sha256": sha256_file(reference_path),
        "environment": _torch_environment(args.device),
        "command": [str(value) for value in sys.argv],
        "config_snapshot": config,
        "expected_cases": dict(expected),
        "executed_cases": dict(executed),
        "dtype": "FP64 exact legal solver / FP32 continuous and output tensors",
        "oracle_only": True,
    }


def _write_stage(
    output_dir: Path,
    result: Mapping[str, Any],
    manifest: Mapping[str, Any],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    (output_dir / "run_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    linear = result["metrics"]["linear"]
    attention = result["metrics"]["attention"]
    lines = [
        "# joint output-gauge probe: J0",
        "",
        f"- status: `{result['status']}`",
        f"- parent SHA256: `{manifest['parent_sha256']}`",
        f"- cache SHA256: `{manifest['cache_sha256']}`",
        f"- tool SHA256: `{manifest['tool_sha256']}`",
        "",
        f"## Gate: {result['gate']}",
        "",
        str(result["reason"]),
        "",
        "## Linear",
        "",
        f"- gate pass: `{linear['gate_pass']}`",
        f"- controls median fold gain: `{linear['control_median_gain']:+.6f}`",
        f"- positive focus layers (>1%): `{linear['positive_focus_layers']}/{linear['layer_count']}`",
        "",
        "| layer | focus fold-median gain | focus positive roles |",
        "|---:|---:|---:|",
    ]
    for item in linear["layer_summary"]:
        lines.append(
            f"| {item['layer']} | {item['focus_median_gain']:+.6f} | "
            f"{item['focus_positive_roles']} |"
        )
    lines.extend([
        "",
        "## Attention",
        "",
        f"- gate pass: `{attention['gate_pass']}`",
        f"- positive joint layers: `{attention['positive_joint_layers']}/{attention['layer_count']}`",
        "",
        "| layer | joint fold-median gain | logit gain | probability gain |",
        "|---:|---:|---:|---:|",
    ])
    for item in attention["layer_summary"]:
        lines.append(
            f"| {item['layer']} | {item['joint_median_gain']:+.6f} | "
            f"{item['joint_logit_median_gain']:+.6f} | "
            f"{item['joint_probability_median_gain']:+.6f} |"
        )
    lines.extend([
        "",
        "All gauges, exact legal replacements, and fold results are calibration-only oracle records.",
    ])
    (output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _exact_decode_many(values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Decode [V,N,B,64] independent legal blocks as [V,N,B,64]."""

    if values.ndim != 4 or int(values.shape[-1]) != 64:
        raise ValueError(f"expected [variants,samples,blocks,64], got {tuple(values.shape)}")
    variants, samples, blocks = map(int, values.shape[:3])
    flat = values.permute(0, 2, 1, 3).reshape(variants * blocks * samples, 64)
    params, loss = exact_legal_blocks(flat)
    decoded = ref.dequantize_hif4(
        params, (variants * blocks * samples, 64)
    ).to(torch.float32)
    decoded = decoded.reshape(variants, blocks, samples, 64).permute(0, 2, 1, 3).contiguous()
    return decoded, loss.reshape(variants, blocks, samples).permute(0, 2, 1).contiguous()


def _single_group_variants(
    values: torch.Tensor,
    gauge_values: Sequence[float] = GAUGE_VALUES,
    group_width: int = 8,
) -> tuple[torch.Tensor, list[dict[str, Any]]]:
    """Create one-gauge-at-a-time variants for every block/group/option."""

    if values.ndim != 3 or int(values.shape[-1]) != 64:
        raise ValueError(f"expected [samples,blocks,64], got {tuple(values.shape)}")
    if 64 % group_width != 0:
        raise ValueError("group width must divide 64")
    samples, blocks = map(int, values.shape[:2])
    group_count = 64 // group_width
    specs = [
        {"block": block, "group": group, "option": option, "gauge": float(gauge)}
        for block in range(blocks)
        for group in range(group_count)
        for option, gauge in enumerate(gauge_values)
    ]
    variants = values.unsqueeze(0).expand(len(specs), samples, blocks, 64).clone()
    for index, spec in enumerate(specs):
        start = int(spec["group"]) * group_width
        stop = start + group_width
        variants[index, :, int(spec["block"]), start:stop] *= float(spec["gauge"])
    return variants, specs


def _paired_group_variants(
    q_values: torch.Tensor,
    k_values: torch.Tensor,
    gauge_values: Sequence[float] = GAUGE_VALUES,
    group_width: int = 8,
) -> tuple[torch.Tensor, torch.Tensor, list[dict[str, Any]]]:
    """Create paired Q*G/K*G^-T one-group-at-a-time variants."""

    q_variants, specs = _single_group_variants(q_values, gauge_values, group_width)
    k_variants = k_values.unsqueeze(0).expand(len(specs), *k_values.shape).clone()
    for index, spec in enumerate(specs):
        start = int(spec["group"]) * group_width
        stop = start + group_width
        k_variants[index, :, int(spec["block"]), start:stop] /= float(spec["gauge"])
    return q_variants, k_variants, specs


def _apply_group_gauge(
    values: torch.Tensor,
    gauges: torch.Tensor,
    inverse: bool,
    group_width: int = 8,
) -> torch.Tensor:
    """Apply [blocks,groups] diagonal gauges to [samples,blocks,64]."""

    if values.ndim != 3 or gauges.ndim != 2:
        raise ValueError("values/gauges have incompatible ranks")
    if int(values.shape[1]) != int(gauges.shape[0]) or int(values.shape[-1]) != 64:
        raise ValueError("values/gauges have incompatible block dimensions")
    if 64 % group_width != 0 or int(gauges.shape[1]) != 64 // group_width:
        raise ValueError("values/gauges have incompatible group dimensions")
    expanded = gauges.repeat_interleave(group_width, dim=1).to(
        device=values.device, dtype=values.dtype
    )
    if inverse:
        expanded = expanded.reciprocal()
    return values * expanded.unsqueeze(0)


def _linear_output(
    activation_full: torch.Tensor,
    weight_full: torch.Tensor,
    activation_selected_continuous: torch.Tensor,
    weight_selected_continuous: torch.Tensor,
    activation_decoded: torch.Tensor,
    weight_decoded: torch.Tensor,
) -> torch.Tensor:
    """Assemble output after replacing selected blocks with decoded values."""

    reference = activation_full @ weight_full.transpose(0, 1)
    continuous_selected = torch.einsum(
        "nbd,rbd->nr", activation_selected_continuous, weight_selected_continuous
    )
    decoded_selected = torch.einsum(
        "nbd,rbd->nr", activation_decoded, weight_decoded
    )
    return reference - continuous_selected + decoded_selected


def _mse(value: torch.Tensor, reference: torch.Tensor) -> float:
    result = float((value.to(torch.float32) - reference.to(torch.float32)).square().mean())
    if not math.isfinite(result):
        raise ValueError("non-finite MSE")
    return result


def _gain(baseline: float, candidate: float) -> float:
    return (baseline - candidate) / max(baseline, EPS)


def _linear_record(
    raw: Any,
    parent_path: Path,
    layer: int,
    role: str,
    args: argparse.Namespace,
    settings: Mapping[str, Any],
    state_cache: dict[int, tuple[Any, dict[tuple[int, str], tuple[Any, dict[str, torch.Tensor]]]]],
) -> dict[str, Any]:
    device = torch.device(args.device)
    state, _, artifact = _load_parent_state(
        raw, parent_path, layer, role, args.device, state_cache
    )
    weight_pair = v2._move_pair(v2._pair(raw.weights[layer][role]), device)
    weight_reference = v2.dequantize_nvfp4(*weight_pair).to(torch.float32)
    weight_continuous = _state_parent_transform(weight_reference, state)
    rows = _uniform_indices(int(weight_continuous.shape[0]), int(settings["weight_rows"]))
    block_count = int(weight_continuous.shape[1]) // 64
    blocks = _uniform_indices(block_count, int(settings["blocks"]))
    row_tensor = torch.as_tensor(rows, dtype=torch.int64, device=device)
    weight_full = weight_continuous.index_select(0, row_tensor)
    weight_values = torch.stack(
        [weight_full[:, block * 64:(block + 1) * 64] for block in blocks], dim=1
    )
    fixed_weight, fixed_weight_loss = _batched_exact_blocks(weight_values)
    candidate_weight_values, candidate_specs = _single_group_variants(
        weight_values,
        tuple(float(value) for value in settings["gauge_values"]),
        int(settings["group_width"]),
    )
    candidate_weight, candidate_weight_loss = _exact_decode_many(candidate_weight_values)

    windows: list[dict[str, Any]] = []
    token_count = int(settings["activation_tokens"])
    for calibration_index in range(len(raw.calibration_windows)):
        activation_pair = v2._move_pair(
            v2._pair(raw.calibration_activations[role][calibration_index][layer]), device
        )
        activation_reference = v2.dequantize_nvfp4(*activation_pair).to(torch.float32)
        activation_continuous = _state_activation_transform(activation_reference, state)
        activation_rows = _uniform_indices(int(activation_continuous.shape[0]), token_count)
        activation_row_tensor = torch.as_tensor(activation_rows, dtype=torch.int64, device=device)
        activation_full = activation_continuous.index_select(0, activation_row_tensor)
        activation_values = torch.stack(
            [activation_full[:, block * 64:(block + 1) * 64] for block in blocks], dim=1
        )
        fixed_activation, fixed_activation_loss = _batched_exact_blocks(activation_values)
        windows.append({
            "index": calibration_index,
            "rows": list(activation_rows),
            "full": activation_full,
            "values": activation_values,
            "fixed": fixed_activation,
            "fixed_loss": fixed_activation_loss,
            "continuous": activation_full,
        })

    fold_records: list[dict[str, Any]] = []
    for holdout, train_indices in _loo_folds(len(windows)):
        train_values = torch.cat([windows[index]["values"] for index in train_indices], dim=0)
        train_full = torch.cat([windows[index]["full"] for index in train_indices], dim=0)
        train_reference = train_full @ weight_full.transpose(0, 1)
        train_candidate_values, _ = _single_group_variants(
            train_values,
            tuple(float(value) for value in settings["gauge_values"]),
            int(settings["group_width"]),
        )
        train_candidate, _ = _exact_decode_many(train_candidate_values)
        train_candidate_outputs = []
        for variant in range(len(candidate_specs)):
            train_candidate_outputs.append(_linear_output(
                train_full,
                weight_full,
                train_values,
                weight_values,
                train_candidate[variant],
                candidate_weight[variant],
            ))
        train_candidate_output = torch.stack(train_candidate_outputs, dim=0)
        train_candidate_mse = (train_candidate_output - train_reference.unsqueeze(0)).square().mean(dim=(1, 2))
        block_count_selected = len(blocks)
        group_count = 64 // int(settings["group_width"])
        train_mse_by_group = train_candidate_mse.reshape(block_count_selected, group_count, -1)
        selected_options = torch.argmin(train_mse_by_group, dim=2)
        gauge_matrix = torch.tensor(
            [[float(settings["gauge_values"][int(selected_options[b, g])]) for g in range(group_count)]
             for b in range(block_count_selected)],
            dtype=torch.float32,
            device=device,
        )

        window = windows[holdout]
        candidate_values = _apply_group_gauge(
            window["values"], gauge_matrix, inverse=True, group_width=int(settings["group_width"])
        )
        candidate_weight_values_fold = _apply_group_gauge(
            weight_values, gauge_matrix, inverse=False, group_width=int(settings["group_width"])
        )
        candidate_activation_fold, candidate_activation_loss = _batched_exact_blocks(candidate_values)
        candidate_weight_fold, candidate_weight_loss_fold = _batched_exact_blocks(candidate_weight_values_fold)
        baseline_output = _linear_output(
            window["full"], weight_full, window["values"], weight_values,
            window["fixed"].permute(1, 0, 2), fixed_weight.permute(1, 0, 2),
        )
        candidate_output = _linear_output(
            window["full"], weight_full, candidate_values, candidate_weight_values_fold,
            candidate_activation_fold.permute(1, 0, 2), candidate_weight_fold.permute(1, 0, 2),
        )
        reference_output = window["full"] @ weight_full.transpose(0, 1)
        baseline_mse = _mse(baseline_output, reference_output)
        candidate_mse = _mse(candidate_output, reference_output)
        baseline_operand_sse = float(
            (window["fixed"] - window["values"].permute(1, 0, 2)).square().sum()
            + (fixed_weight - weight_values.permute(1, 0, 2)).square().sum()
        )
        candidate_operand_sse = float(
            (candidate_activation_fold - candidate_values.permute(1, 0, 2)).square().sum()
            + (candidate_weight_fold - candidate_weight_values_fold.permute(1, 0, 2)).square().sum()
        )
        fold_records.append({
            "holdout_window": holdout,
            "train_windows": list(train_indices),
            "selected_options": selected_options.cpu().tolist(),
            "selected_gauges": gauge_matrix.cpu().tolist(),
            "selection_min_variant_mse": float(train_candidate_mse.min()),
            "baseline": {
                "output_mse": baseline_mse,
                "operand_sse": baseline_operand_sse,
                "solver_loss_activation": float(window["fixed_loss"].sum()),
                "solver_loss_weight": float(fixed_weight_loss.sum()),
            },
            "candidate": {
                "output_mse": candidate_mse,
                "operand_sse": candidate_operand_sse,
                "output_gain": _gain(baseline_mse, candidate_mse),
                "solver_loss_activation": float(candidate_activation_loss.sum()),
                "solver_loss_weight": float(candidate_weight_loss_fold.sum()),
            },
            "finite": True,
        })

    fold_gains = [float(item["candidate"]["output_gain"]) for item in fold_records]
    return {
        "layer": layer,
        "role": role,
        "shape": list(weight_reference.shape),
        "rows": list(rows),
        "blocks": list(blocks),
        "group_width": int(settings["group_width"]),
        "gauge_values": [float(value) for value in settings["gauge_values"]],
        "calibration_windows": list(range(len(windows))),
        "parent_calibration_artifact": artifact,
        "folds": fold_records,
        "median_output_gain": float(torch.median(torch.tensor(fold_gains, dtype=torch.float64))),
        "mean_output_gain": sum(fold_gains) / max(len(fold_gains), 1),
        "worst_output_gain": min(fold_gains),
        "finite": True,
    }


def _attention_trace_metrics(
    q: torch.Tensor,
    k: torch.Tensor,
    value: torch.Tensor,
    raw: Any,
    reference_output: torch.Tensor,
    reference_logits: torch.Tensor,
    reference_probabilities: torch.Tensor,
) -> dict[str, float]:
    output, logits, probabilities = v2._attention_trace(
        q, k, value, raw.q_heads, raw.kv_heads, raw.head_dim
    )
    return {
        "output_mse": _mse(output, reference_output),
        "logit_mse": _mse(logits, reference_logits),
        "probability_mse": _mse(probabilities, reference_probabilities),
    }


def _attention_window(
    raw: Any,
    states: Mapping[str, Any],
    layer: int,
    calibration_index: int,
    settings: Mapping[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    q_raw, k_raw, v_raw = raw.calibration_qkv[calibration_index][layer]
    q_pair = v2._move_pair(v2._pair(q_raw), device)
    k_pair = v2._move_pair(v2._pair(k_raw), device)
    v_pair = v2._move_pair(v2._pair(v_raw), device)
    q_reference = v2.dequantize_nvfp4(*q_pair).to(torch.float32)
    k_reference = v2.dequantize_nvfp4(*k_pair).to(torch.float32)
    v_reference = v2.dequantize_nvfp4(*v_pair).to(torch.float32)
    q_parent_params = sol.hif4_dynamic_quantize_q(
        q_pair[0], q_pair[1], raw.q_heads, raw.head_dim, states["q_state"]
    )
    k_parent_params = sol.hif4_dynamic_quantize_k(
        k_pair[0], k_pair[1], raw.kv_heads, raw.head_dim, states["k_state"]
    )
    v_parent_params = sol.hif4_dynamic_quantize_v(
        v_pair[0], v_pair[1], raw.kv_heads, raw.head_dim, states["v_state"]
    )
    q_parent = ref.dequantize_hif4(q_parent_params, tuple(q_reference.shape)).to(torch.float32)
    k_parent = ref.dequantize_hif4(k_parent_params, tuple(k_reference.shape)).to(torch.float32)
    v_parent = ref.dequantize_hif4(v_parent_params, tuple(v_reference.shape)).to(torch.float32)
    q_continuous = sol._attention_state_transform_dense(
        q_reference, states["q_state"], raw.q_heads, raw.head_dim, is_k=False
    )
    k_continuous = sol._attention_state_transform_dense(
        k_reference, states["k_state"], raw.kv_heads, raw.head_dim, is_k=True
    )
    tokens = min(int(q_reference.shape[0]), int(k_reference.shape[0]), int(v_reference.shape[0]))
    rows = _uniform_indices(tokens, int(settings["tokens"]))
    row_tensor = torch.as_tensor(rows, dtype=torch.int64, device=device)
    q_head = int(settings["q_head"])
    kv_head = raw.kv_heads - 1 if str(settings["kv_head"]) == "last" else int(settings["kv_head"])
    q_lo = q_head * raw.head_dim
    k_lo = kv_head * raw.head_dim
    q_reference_sample = q_reference.index_select(0, row_tensor)
    k_reference_sample = k_reference.index_select(0, row_tensor)
    v_reference_sample = v_reference.index_select(0, row_tensor)
    q_parent_sample = q_parent.index_select(0, row_tensor)
    k_parent_sample = k_parent.index_select(0, row_tensor)
    v_parent_sample = v_parent.index_select(0, row_tensor)
    q_values = q_continuous.index_select(0, row_tensor)[:, q_lo:q_lo + raw.head_dim].unsqueeze(1)
    k_values = k_continuous.index_select(0, row_tensor)[:, k_lo:k_lo + raw.head_dim].unsqueeze(1)
    q_fixed, q_fixed_loss = _batched_exact_blocks(q_values)
    k_fixed, k_fixed_loss = _batched_exact_blocks(k_values)
    reference_output, reference_logits, reference_probabilities = v2._attention_trace(
        q_reference_sample.unsqueeze(0), k_reference_sample.unsqueeze(0), v_reference_sample.unsqueeze(0),
        raw.q_heads, raw.kv_heads, raw.head_dim,
    )
    q_parent_metrics = _attention_trace_metrics(
        q_parent_sample.unsqueeze(0), k_parent_sample.unsqueeze(0), v_parent_sample.unsqueeze(0), raw,
        reference_output, reference_logits, reference_probabilities,
    )
    return {
        "index": calibration_index,
        "rows": list(rows),
        "q_parent": q_parent_sample,
        "k_parent": k_parent_sample,
        "v_parent": v_parent_sample,
        "q_reference": q_reference_sample,
        "k_reference": k_reference_sample,
        "v_reference": v_reference_sample,
        "q_values": q_values,
        "k_values": k_values,
        "q_fixed": q_fixed,
        "k_fixed": k_fixed,
        "q_fixed_loss": q_fixed_loss,
        "k_fixed_loss": k_fixed_loss,
        "reference_output": reference_output,
        "reference_logits": reference_logits,
        "reference_probabilities": reference_probabilities,
        "parent_metrics": q_parent_metrics,
        "q_head": q_head,
        "kv_head": kv_head,
        "q_lo": q_lo,
        "k_lo": k_lo,
    }


def _attention_record(
    raw: Any,
    parent_path: Path,
    layer: int,
    args: argparse.Namespace,
    settings: Mapping[str, Any],
    state_cache: dict[int, tuple[Any, dict[int, dict[str, Any]]]],
) -> dict[str, Any]:
    device = torch.device(args.device)
    states, artifact = _load_parent_attention_state(raw, parent_path, layer, args.device, state_cache)
    windows = [
        _attention_window(raw, states, layer, index, settings, device)
        for index in range(len(raw.calibration_qkv))
    ]
    gauge_values = tuple(float(value) for value in settings["gauge_values"])
    group_width = int(settings["group_width"])
    fold_records: list[dict[str, Any]] = []
    for holdout, train_indices in _loo_folds(len(windows)):
        candidate_sse: torch.Tensor | None = None
        element_count = 0
        for train_index in train_indices:
            window = windows[train_index]
            q_variants, k_variants, specs = _paired_group_variants(
                window["q_values"], window["k_values"], gauge_values, group_width
            )
            q_decoded, _ = _exact_decode_many(q_variants)
            k_decoded, _ = _exact_decode_many(k_variants)
            variants = int(q_decoded.shape[0])
            samples = int(q_decoded.shape[1])
            q_batch = window["q_parent"].unsqueeze(0).expand(variants, -1, -1).clone()
            k_batch = window["k_parent"].unsqueeze(0).expand(variants, -1, -1).clone()
            v_batch = window["v_parent"].unsqueeze(0).expand(variants, -1, -1)
            q_rows = torch.arange(samples, dtype=torch.int64, device=device)
            k_rows = torch.arange(samples, dtype=torch.int64, device=device)
            q_batch[:, q_rows, window["q_lo"]:window["q_lo"] + raw.head_dim] = q_decoded[:, :, 0]
            k_batch[:, k_rows, window["k_lo"]:window["k_lo"] + raw.head_dim] = k_decoded[:, :, 0]
            output, _, _ = v2._attention_trace(
                q_batch, k_batch, v_batch, raw.q_heads, raw.kv_heads, raw.head_dim
            )
            errors = (output - window["reference_output"]).square().sum(dim=(1, 2))
            candidate_sse = errors if candidate_sse is None else candidate_sse + errors
            element_count += int(output.shape[1] * output.shape[2])
        if candidate_sse is None:
            raise ValueError("empty attention training fold")
        candidate_mse = candidate_sse / max(element_count, 1)
        block_count = 1
        group_count = 64 // group_width
        selected_options = torch.argmin(candidate_mse.reshape(block_count, group_count, -1), dim=2)
        gauge_matrix = torch.tensor(
            [[float(gauge_values[int(selected_options[0, group])]) for group in range(group_count)]],
            dtype=torch.float32,
            device=device,
        )
        window = windows[holdout]
        candidate_q_values = _apply_group_gauge(
            window["q_values"], gauge_matrix, inverse=False, group_width=group_width
        )
        candidate_k_values = _apply_group_gauge(
            window["k_values"], gauge_matrix, inverse=True, group_width=group_width
        )
        candidate_q, candidate_q_loss = _batched_exact_blocks(candidate_q_values)
        candidate_k, candidate_k_loss = _batched_exact_blocks(candidate_k_values)
        q_fixed_full = window["q_parent"].clone()
        q_candidate_full = window["q_parent"].clone()
        k_fixed_full = window["k_parent"].clone()
        k_candidate_full = window["k_parent"].clone()
        q_fixed_full[:, window["q_lo"]:window["q_lo"] + raw.head_dim] = window["q_fixed"][0]
        q_candidate_full[:, window["q_lo"]:window["q_lo"] + raw.head_dim] = candidate_q[0]
        k_fixed_full[:, window["k_lo"]:window["k_lo"] + raw.head_dim] = window["k_fixed"][0]
        k_candidate_full[:, window["k_lo"]:window["k_lo"] + raw.head_dim] = candidate_k[0]
        q_fixed_metrics = _attention_trace_metrics(
            q_fixed_full.unsqueeze(0), window["k_parent"].unsqueeze(0), window["v_parent"].unsqueeze(0), raw,
            window["reference_output"], window["reference_logits"], window["reference_probabilities"],
        )
        q_candidate_metrics = _attention_trace_metrics(
            q_candidate_full.unsqueeze(0), window["k_parent"].unsqueeze(0), window["v_parent"].unsqueeze(0), raw,
            window["reference_output"], window["reference_logits"], window["reference_probabilities"],
        )
        k_fixed_metrics = _attention_trace_metrics(
            window["q_parent"].unsqueeze(0), k_fixed_full.unsqueeze(0), window["v_parent"].unsqueeze(0), raw,
            window["reference_output"], window["reference_logits"], window["reference_probabilities"],
        )
        k_candidate_metrics = _attention_trace_metrics(
            window["q_parent"].unsqueeze(0), k_candidate_full.unsqueeze(0), window["v_parent"].unsqueeze(0), raw,
            window["reference_output"], window["reference_logits"], window["reference_probabilities"],
        )
        joint_fixed_metrics = _attention_trace_metrics(
            q_fixed_full.unsqueeze(0), k_fixed_full.unsqueeze(0), window["v_parent"].unsqueeze(0), raw,
            window["reference_output"], window["reference_logits"], window["reference_probabilities"],
        )
        joint_candidate_metrics = _attention_trace_metrics(
            q_candidate_full.unsqueeze(0), k_candidate_full.unsqueeze(0), window["v_parent"].unsqueeze(0), raw,
            window["reference_output"], window["reference_logits"], window["reference_probabilities"],
        )
        fold_records.append({
            "holdout_window": holdout,
            "train_windows": list(train_indices),
            "selected_options": selected_options.cpu().tolist(),
            "selected_gauges": gauge_matrix.cpu().tolist(),
            "selection_min_variant_mse": float(candidate_mse.min()),
            "q_fixed": {
                **q_fixed_metrics,
                "output_gain_vs_parent": _gain(window["parent_metrics"]["output_mse"], q_fixed_metrics["output_mse"]),
                "solver_loss": float(window["q_fixed_loss"].sum()),
            },
            "q_candidate": {
                **q_candidate_metrics,
                "output_gain_vs_fixed": _gain(q_fixed_metrics["output_mse"], q_candidate_metrics["output_mse"]),
                "solver_loss": float(candidate_q_loss.sum()),
            },
            "k_fixed": {
                **k_fixed_metrics,
                "output_gain_vs_parent": _gain(window["parent_metrics"]["output_mse"], k_fixed_metrics["output_mse"]),
                "solver_loss": float(window["k_fixed_loss"].sum()),
            },
            "k_candidate": {
                **k_candidate_metrics,
                "output_gain_vs_fixed": _gain(k_fixed_metrics["output_mse"], k_candidate_metrics["output_mse"]),
                "solver_loss": float(candidate_k_loss.sum()),
            },
            "joint_fixed": joint_fixed_metrics,
            "joint_candidate": {
                **joint_candidate_metrics,
                "output_gain": _gain(joint_fixed_metrics["output_mse"], joint_candidate_metrics["output_mse"]),
                "logit_gain": _gain(joint_fixed_metrics["logit_mse"], joint_candidate_metrics["logit_mse"]),
                "probability_gain": _gain(joint_fixed_metrics["probability_mse"], joint_candidate_metrics["probability_mse"]),
            },
            "finite": True,
        })

    joint_gains = [float(item["joint_candidate"]["output_gain"]) for item in fold_records]
    logit_gains = [float(item["joint_candidate"]["logit_gain"]) for item in fold_records]
    probability_gains = [float(item["joint_candidate"]["probability_gain"]) for item in fold_records]
    return {
        "layer": layer,
        "q_head": int(settings["q_head"]),
        "kv_head": windows[0]["kv_head"] if windows else None,
        "group_width": group_width,
        "gauge_values": list(gauge_values),
        "parent_calibration_artifact": artifact,
        "folds": fold_records,
        "median_joint_output_gain": float(torch.median(torch.tensor(joint_gains, dtype=torch.float64))),
        "median_joint_logit_gain": float(torch.median(torch.tensor(logit_gains, dtype=torch.float64))),
        "median_joint_probability_gain": float(torch.median(torch.tensor(probability_gains, dtype=torch.float64))),
        "finite": True,
    }


def _median(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return float(torch.median(torch.tensor(list(values), dtype=torch.float64)))


def run_j0(
    args: argparse.Namespace,
    config: Mapping[str, Any],
    config_path: Path,
    output_dir: Path,
    parent_path: Path,
    cache_path: Path,
) -> dict[str, Any]:
    started = time.perf_counter()
    raw = v2.load_pack(cache_path)
    layers = [int(value) for value in config["layers"]]
    linear_settings = config["linear"]
    attention_settings = config["attention"]
    roles = [str(value) for value in linear_settings["roles"]]
    focus_roles = {str(value) for value in linear_settings["focus_roles"]}
    prepared_cache: dict[int, tuple[Any, dict[tuple[int, str], tuple[Any, dict[str, torch.Tensor]]]]] = {}
    linear_records: list[dict[str, Any]] = []
    artifact_paths: set[str] = set()
    with torch.inference_mode():
        for layer in layers:
            for role in roles:
                record = _linear_record(
                    raw, parent_path, layer, role, args, linear_settings, prepared_cache
                )
                linear_records.append(record)
                artifact_paths.add(record["parent_calibration_artifact"])

    attention_cache: dict[int, tuple[Any, dict[int, dict[str, Any]]]] = {}
    attention_records: list[dict[str, Any]] = []
    with torch.inference_mode():
        for layer in layers:
            record = _attention_record(
                raw, parent_path, layer, args, attention_settings, attention_cache
            )
            attention_records.append(record)
            artifact_paths.add(record["parent_calibration_artifact"])

    gate_settings = config["gate"]
    linear_layer_summary: list[dict[str, Any]] = []
    for layer in layers:
        focus = [item for item in linear_records if item["layer"] == layer and item["role"] in focus_roles]
        all_records = [item for item in linear_records if item["layer"] == layer]
        focus_medians = [float(item["median_output_gain"]) for item in focus]
        all_medians = [float(item["median_output_gain"]) for item in all_records]
        linear_layer_summary.append({
            "layer": layer,
            "focus_median_gain": _median(focus_medians),
            "focus_positive_roles": sum(value > 0.0 for value in focus_medians),
            "focus_role_count": len(focus),
            "all_role_median_gain": _median(all_medians),
        })
    focus_layer_threshold = float(gate_settings["linear_focus_layer_gain"])
    positive_focus_layers = [
        item["layer"] for item in linear_layer_summary
        if item["focus_median_gain"] > focus_layer_threshold
    ]
    controls = [
        float(item["median_output_gain"])
        for item in linear_records if item["role"] not in focus_roles
    ]
    control_median = _median(controls)
    linear_gate = {
        "gate_pass": (
            len(positive_focus_layers) >= int(gate_settings["linear_positive_layers"])
            and any(int(layer) < 15 for layer in positive_focus_layers)
            and any(int(layer) >= 15 for layer in positive_focus_layers)
            and control_median >= float(gate_settings["linear_control_floor"])
        ),
        "focus_layer_median_gain": {
            str(item["layer"]): item["focus_median_gain"] for item in linear_layer_summary
        },
        "positive_focus_layers": len(positive_focus_layers),
        "positive_focus_layer_ids": positive_focus_layers,
        "layer_count": len(layers),
        "control_median_gain": control_median,
        "threshold": focus_layer_threshold,
        "control_floor": float(gate_settings["linear_control_floor"]),
        "layer_summary": linear_layer_summary,
    }

    attention_layer_summary: list[dict[str, Any]] = []
    for record in attention_records:
        attention_layer_summary.append({
            "layer": record["layer"],
            "joint_median_gain": record["median_joint_output_gain"],
            "joint_logit_median_gain": record["median_joint_logit_gain"],
            "joint_probability_median_gain": record["median_joint_probability_gain"],
            "logit_probability_not_both_worse": not (
                record["median_joint_logit_gain"] < 0.0
                and record["median_joint_probability_gain"] < 0.0
            ),
        })
    positive_joint_layers = [
        item["layer"] for item in attention_layer_summary if item["joint_median_gain"] > 0.0
    ]
    attention_gate = {
        "gate_pass": (
            len(positive_joint_layers) >= int(gate_settings["attention_positive_layers"])
            and all(item["logit_probability_not_both_worse"] for item in attention_layer_summary)
        ),
        "positive_joint_layers": len(positive_joint_layers),
        "positive_joint_layer_ids": positive_joint_layers,
        "layer_count": len(layers),
        "layer_summary": attention_layer_summary,
    }

    linear_pass = bool(linear_gate["gate_pass"])
    attention_pass = bool(attention_gate["gate_pass"])
    if linear_pass and attention_pass:
        status = "PASS_TO_J1"
        gate = "J0_PASS_TO_J1"
        next_step = "j1"
    elif linear_pass or attention_pass:
        status = "PARTIAL_PASS"
        gate = "J0_PARTIAL_PASS_TO_SIDE_SPECIFIC_J1"
        next_step = "j1-linear" if linear_pass else "j1-attention"
    else:
        status = "REJECTED"
        gate = "NO_SUPPORTED_JOINT_OUTPUT_GAUGE_ORACLE"
        next_step = "close-plan"

    expected = {
        "layers": layers,
        "linear_records": len(layers) * len(roles),
        "attention_records": len(layers),
        "calibration_folds": len(raw.calibration_windows),
        "linear_focus_roles": sorted(focus_roles),
        "linear_controls": sorted(set(roles) - focus_roles),
        "gauge_exponents": list(GAUGE_EXPONENTS),
        "objective": "LOO actual output MSE after legal paired group gauge",
    }
    executed = {
        "linear_records": len(linear_records),
        "attention_records": len(attention_records),
        "linear_loo_folds": len(linear_records) * len(raw.calibration_windows),
        "attention_loo_folds": len(attention_records) * len(raw.calibration_windows),
        "linear_candidate_variants_per_record": int(linear_settings["blocks"]) * (64 // int(linear_settings["group_width"])) * len(linear_settings["gauge_values"]),
        "attention_candidate_variants_per_record": (64 // int(attention_settings["group_width"])) * len(attention_settings["gauge_values"]),
        "parent_artifacts": sorted(artifact_paths),
        "elapsed_s": time.perf_counter() - started,
    }
    result = {
        "stage": "j0",
        "status": status,
        "hypothesis": "同一输出产品中的 X/W 或 Q/K 共同选择网格坐标，可能降低最终输出误差。",
        "expected_cases": expected,
        "executed_cases": executed,
        "failed_cases": [],
        "metrics": {
            "linear": {
                **linear_gate,
                "records": linear_records,
            },
            "attention": {
                **attention_gate,
                "records": attention_records,
            },
            "elapsed_s": time.perf_counter() - started,
        },
        "gate": gate,
        "reason": (
            "J0 oracle met both fixed side gates; a separate J1 plan may register one fixed statistical deployment rule."
            if status == "PASS_TO_J1" else
            "J0 met one fixed side gate; only that side may receive a separate J1 plan."
            if status == "PARTIAL_PASS" else
            "The fixed LOO joint output-gauge oracle did not meet either side's material-gain gate; close this plan without scanning the gauge family."
        ),
        "next_step": next_step,
    }
    manifest = _manifest(args, config_path, cache_path, parent_path, expected, executed, status)
    _write_stage(output_dir, result, manifest)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("j0",), required=True)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser


def main(args: argparse.Namespace | None = None) -> dict[str, Any]:
    args = args or build_parser().parse_args()
    parent_path = args.parent.resolve()
    cache_path = args.cache.resolve()
    config_path = args.config.resolve()
    output_dir = args.output_dir.resolve()
    for path in (parent_path, cache_path, config_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    config = _load_config(config_path)
    return run_j0(args, config, config_path, output_dir, parent_path, cache_path)


if __name__ == "__main__":
    result = main()
    print(json.dumps({
        "stage": result["stage"],
        "status": result["status"],
        "gate": result["gate"],
        "next_step": result["next_step"],
    }, ensure_ascii=False))
