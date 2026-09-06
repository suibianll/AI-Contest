"""Offline D-A/D-B diagnostics for the v186 HiF4 parent.

The tool deliberately stays outside ``solution.py``.  D-A describes the
activation error anatomy of the existing dynamic path.  D-B compares the
current legal 64-block partition with one fixed, shared-pressure permutation
oracle.  Both stages use calibration data only and never create a deployable
state or candidate solution.
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
from legal_codec_output_probe import (  # noqa: E402
    _load_parent_attention_state,
    _load_parent_state,
    _state_activation_transform,
    _state_parent_transform,
    exact_legal_blocks,
    sha256_file,
)


EPS = 1.0e-12
SHARD_COUNT = int(v3.SHARD_COUNT)


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


def _load_config(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("config must be an object")
    return value


def _manifest(
    stage: str,
    args: argparse.Namespace,
    config_path: Path,
    cache_path: Path,
    parent_path: Path,
    expected: Mapping[str, Any],
    executed: Mapping[str, Any],
    status: str,
) -> dict[str, Any]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    return {
        "stage": stage,
        "status": status,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "git_head": _git_head(),
        "parent": str(parent_path.resolve()),
        "parent_sha256": sha256_file(parent_path),
        "tool": str(Path(__file__).resolve()),
        "tool_sha256": sha256_file(Path(__file__).resolve()),
        "config": str(config_path.resolve()),
        "config_sha256": sha256_file(config_path),
        "cache": str(cache_path.resolve()),
        "cache_sha256": sha256_file(cache_path),
        "reference_hif4": str((ROOT / "evaluator" / "reference_hif4.py").resolve()),
        "reference_hif4_sha256": sha256_file(ROOT / "evaluator" / "reference_hif4.py"),
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
    lines = [
        f"# hierarchy/activation probe: {result['stage']}",
        "",
        f"- status: `{result['status']}`",
        f"- parent SHA256: `{manifest['parent_sha256']}`",
        f"- cache SHA256: `{manifest['cache_sha256']}`",
        f"- expected cases: `{json.dumps(result['expected_cases'], ensure_ascii=False, sort_keys=True)}`",
        f"- executed cases: `{json.dumps(result['executed_cases'], ensure_ascii=False, sort_keys=True)}`",
        "",
        f"## Gate: {result['gate']}",
        "",
        str(result["reason"]),
        "",
    ]
    if result["stage"] == "d-a":
        metrics = result["metrics"]
        lines.extend([
            "## D-A summary",
            "",
            f"- records: `{metrics['record_count']}`",
            f"- clip element fraction: `{metrics['clip_element_fraction']:.6f}`",
            f"- clip block fraction: `{metrics['clip_block_fraction']:.6f}`",
            f"- clip SSE fraction of nearest-grid SSE: `{metrics['clip_sse_fraction']:.6f}`",
            f"- parent transformed-coordinate MSE: `{metrics['parent_mse']:.6e}`",
            "",
            "| layer | role | window | ratio p50 | ratio p90 | ratio p99 | clip elements | clip SSE fraction | parent MSE |",
            "|---:|---|---:|---:|---:|---:|---:|---:|---:|",
        ])
        for item in metrics["records"]:
            lines.append(
                f"| {item['layer']} | {item['role']} | {item['calibration_window']} | "
                f"{item['ratio']['p50']:.4f} | {item['ratio']['p90']:.4f} | {item['ratio']['p99']:.4f} | "
                f"{item['clip']['element_fraction']:.4f} | {item['clip']['sse_fraction']:.4f} | "
                f"{item['parent']['mse']:.3e} |"
            )
    elif result["stage"] == "d-b":
        gate = result["metrics"]["gate"]
        lines.extend([
            "## D-B fixed gate",
            "",
            f"- focus layer median gains: `{json.dumps(gate['focus_layer_median_gain'], ensure_ascii=False, sort_keys=True)}`",
            f"- positive focus layers (>1%): `{gate['positive_focus_layers']}/{gate['layer_count']}`",
            f"- deep positive: `{gate['deep_positive']}`; non-deep positive: `{gate['non_deep_positive']}`",
            "",
            "### Linear layer summary",
            "",
            "| layer | focus median output gain | focus positive roles | all-role median output gain |",
            "|---:|---:|---:|---:|",
        ])
        for item in result["metrics"]["linear_layer_summary"]:
            lines.append(
                f"| {item['layer']} | {item['focus_median_output_gain']:+.6f} | "
                f"{item['focus_positive_roles']} | {item['all_role_median_output_gain']:+.6f} |"
            )
        lines.extend([
            "",
            "### Attention oracle summary",
            "",
            f"- Q-only median gain versus fixed partition: `{result['metrics']['attention_summary']['q_median_gain']:+.6f}`",
            f"- K-only median gain versus fixed partition: `{result['metrics']['attention_summary']['k_median_gain']:+.6f}`",
            f"- joint median gain versus fixed partition: `{result['metrics']['attention_summary']['joint_median_gain']:+.6f}`",
            "",
            "The permutation and all legal exact replacements are calibration-only oracle records.",
        ])
    (output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _metric(reference: torch.Tensor, value: torch.Tensor) -> dict[str, float]:
    error = (value.to(torch.float32) - reference.to(torch.float32)).square()
    return {"sse": float(error.sum()), "mse": float(error.mean())}


def _quantiles(value: torch.Tensor) -> dict[str, float]:
    flat = value.detach().to(torch.float32).reshape(-1)
    if flat.numel() == 0 or not bool(torch.isfinite(flat).all()):
        raise ValueError("quantile input is empty or non-finite")
    points = torch.tensor((0.50, 0.90, 0.99), dtype=flat.dtype, device=flat.device)
    result = torch.quantile(flat, points)
    return {"p50": float(result[0]), "p90": float(result[1]), "p99": float(result[2])}


def _effective_activation_unit(
    params: Mapping[str, torch.Tensor],
    shape: Sequence[int],
    device: torch.device,
) -> torch.Tensor:
    tokens, width = map(int, shape)
    if width % 64 != 0:
        raise ValueError(f"activation width is not divisible by 64: {width}")
    blocks = width // 64
    scale = params["scale_factor"].to(device=device, dtype=torch.float32).reshape(tokens, blocks)
    lv2 = params["scale_lv2"].to(device=device, dtype=torch.float32).reshape(tokens, blocks, 8)
    lv3 = params["scale_lv3"].to(device=device, dtype=torch.float32).reshape(tokens, blocks, 8, 2)
    return (
        scale[:, :, None, None, None]
        * lv2[:, :, :, None, None]
        * lv3[:, :, :, :, None]
        * 0.25
    ).expand(tokens, blocks, 8, 2, 4).reshape(tokens, blocks, 64).reshape(tokens, width)


def _validate_finite(name: str, value: torch.Tensor) -> None:
    if not bool(torch.isfinite(value.to(torch.float32)).all()):
        raise ValueError(f"{name} contains non-finite values")


def _load_raw_and_state(
    raw: Any,
    parent_path: Path,
    layer: int,
    role: str,
    device: str,
    cache: dict[int, tuple[Any, dict[tuple[int, str], tuple[Any, dict[str, torch.Tensor]]]]],
) -> tuple[dict[str, Any], dict[str, torch.Tensor], str]:
    state, params, artifact = _load_parent_state(raw, parent_path, layer, role, device, cache)
    return dict(state), dict(params), artifact


def run_d_a(
    args: argparse.Namespace,
    config: Mapping[str, Any],
    config_path: Path,
    output_dir: Path,
    parent_path: Path,
    cache_path: Path,
) -> dict[str, Any]:
    started = time.perf_counter()
    device = torch.device(args.device)
    raw = v2.load_pack(cache_path)
    settings = config["linear"]
    layers = [int(value) for value in config["layers"]]
    roles = [str(value) for value in settings["roles"]]
    token_count = int(settings["activation_tokens"])
    calibration_windows = list(range(len(raw.calibration_windows)))
    prepared_cache: dict[int, tuple[Any, dict[tuple[int, str], tuple[Any, dict[str, torch.Tensor]]]]] = {}
    records: list[dict[str, Any]] = []
    clip_elements = 0
    total_elements = 0
    clip_blocks = 0
    total_blocks = 0
    clip_sse = 0.0
    grid_sse = 0.0
    parent_sse = 0.0

    with torch.inference_mode():
        for layer in layers:
            for role in roles:
                if role not in raw.weights[layer]:
                    raise ValueError(f"missing Linear role {role} at layer {layer}")
                state, _, artifact = _load_raw_and_state(
                    raw, parent_path, layer, role, args.device, prepared_cache
                )
                for calibration_index in calibration_windows:
                    pair = v2._move_pair(
                        v2._pair(raw.calibration_activations[role][calibration_index][layer]),
                        device,
                    )
                    reference = v2.dequantize_nvfp4(*pair).to(torch.float32)
                    params = sol.hif4_dynamic_quantize_activation(pair[0], pair[1], state)
                    ref.validate_hif4_params(params, tuple(reference.shape))
                    transformed = _state_activation_transform(reference, state)
                    decoded = ref.dequantize_hif4(params, tuple(reference.shape)).to(torch.float32)
                    unit = _effective_activation_unit(params, reference.shape, device)
                    selected_rows = _uniform_indices(int(reference.shape[0]), token_count)
                    selected = torch.as_tensor(selected_rows, dtype=torch.int64, device=device)
                    transformed = transformed[selected]
                    decoded = decoded[selected]
                    unit = unit[selected]
                    ratio = transformed.abs() / unit.clamp_min(EPS)
                    clip_mask = ratio > 7.0
                    nearest = torch.sign(transformed) * ratio.round().clamp(0.0, 7.0) * unit
                    nearest_error = (nearest - transformed).square()
                    parent_error = (decoded - transformed).square()
                    _validate_finite("activation transformed", transformed)
                    _validate_finite("activation effective unit", unit)
                    _validate_finite("activation ratio", ratio)
                    _validate_finite("activation parent decode", decoded)
                    current_elements = int(ratio.numel())
                    current_blocks = int(ratio.shape[0] * ratio.shape[1] // 64)
                    current_clip_elements = int(clip_mask.sum())
                    current_clip_blocks = int(clip_mask.reshape(ratio.shape[0], -1, 64).any(dim=-1).sum())
                    current_grid_sse = float(nearest_error.sum())
                    current_clip_sse = float(nearest_error.masked_select(clip_mask).sum())
                    current_parent_sse = float(parent_error.sum())
                    total_elements += current_elements
                    clip_elements += current_clip_elements
                    total_blocks += current_blocks
                    clip_blocks += current_clip_blocks
                    grid_sse += current_grid_sse
                    clip_sse += current_clip_sse
                    parent_sse += current_parent_sse
                    records.append({
                        "layer": layer,
                        "role": role,
                        "calibration_window": calibration_index,
                        "shape": list(reference.shape),
                        "tokens": int(reference.shape[0]),
                        "sample_rows": list(selected_rows),
                        "parent_calibration_artifact": artifact,
                        "effective_unit": _quantiles(unit),
                        "ratio": _quantiles(ratio),
                        "clip": {
                            "element_count": current_clip_elements,
                            "element_fraction": current_clip_elements / max(current_elements, 1),
                            "block_count": current_clip_blocks,
                            "block_fraction": current_clip_blocks / max(current_blocks, 1),
                            "sse": current_clip_sse,
                            "sse_fraction": current_clip_sse / max(current_grid_sse, EPS),
                        },
                        "grid": {
                            "sse": current_grid_sse,
                            "mse": current_grid_sse / max(current_elements, 1),
                            "non_clip_sse": current_grid_sse - current_clip_sse,
                        },
                        "parent": {
                            "sse": current_parent_sse,
                            "mse": current_parent_sse / max(current_elements, 1),
                        },
                        "finite": True,
                    })

    metrics = {
        "record_count": len(records),
        "clip_element_fraction": clip_elements / max(total_elements, 1),
        "clip_block_fraction": clip_blocks / max(total_blocks, 1),
        "clip_sse_fraction": clip_sse / max(grid_sse, EPS),
        "clip_sse": clip_sse,
        "grid_sse": grid_sse,
        "parent_sse": parent_sse,
        "parent_mse": parent_sse / max(total_elements, 1),
        "records": records,
    }
    expected = {
        "layers": layers,
        "roles": roles,
        "calibration_windows": len(calibration_windows),
        "activation_tokens_requested": token_count,
        "objective": "activation final-coordinate ratio and nearest-grid clip decomposition",
    }
    executed = {
        "records": len(records),
        "activation_api_calls": len(records),
        "total_elements": total_elements,
        "total_blocks": total_blocks,
        "parent_artifacts": sorted({item["parent_calibration_artifact"] for item in records}),
        "elapsed_s": time.perf_counter() - started,
    }
    result = {
        "stage": "d-a",
        "status": "PASS",
        "hypothesis": "v186 activation error is either clip/long-tail dominated or ordinary legal grid dominated.",
        "expected_cases": expected,
        "executed_cases": executed,
        "failed_cases": [],
        "metrics": metrics,
        "gate": "D_A_DIAGNOSTIC_COMPLETE",
        "reason": "D-A completed on all fixed layers, roles and calibration windows; it is descriptive and does not create a deployable rule.",
        "next_step": "d-b",
    }
    manifest = _manifest("d-a", args, config_path, cache_path, parent_path, expected, executed, result["status"])
    _write_stage(output_dir, result, manifest)
    return result


def _pressure_order(
    activation_values: torch.Tensor,
    weight_values: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    activation_rms = activation_values.to(torch.float32).square().mean(dim=0).sqrt()
    weight_rms = weight_values.to(torch.float32).square().mean(dim=0).sqrt()
    activation_scale = torch.median(activation_rms).clamp_min(EPS)
    weight_scale = torch.median(weight_rms).clamp_min(EPS)
    pressure = torch.maximum(
        torch.log2((activation_rms / activation_scale).clamp_min(EPS)),
        torch.log2((weight_rms / weight_scale).clamp_min(EPS)),
    )
    order = torch.argsort(pressure, descending=True)
    return order, activation_rms, weight_rms


def _decode_exact(
    params: Mapping[str, torch.Tensor],
    batch_count: int,
) -> torch.Tensor:
    decoded = ref.dequantize_hif4(params, (batch_count, 64)).to(torch.float32)
    if tuple(decoded.shape) != (batch_count, 64):
        raise ValueError(f"exact decode shape mismatch: {tuple(decoded.shape)}")
    return decoded


def _restore_permuted_blocks(
    decoded_permuted: torch.Tensor,
    orders: Sequence[torch.Tensor],
) -> torch.Tensor:
    block_count, batch_count, width = (
        int(decoded_permuted.shape[0]),
        int(decoded_permuted.shape[1]),
        int(decoded_permuted.shape[2]),
    )
    if width != 64 or len(orders) != block_count:
        raise ValueError("permuted block shape/order mismatch")
    restored = torch.empty_like(decoded_permuted)
    for block, order in enumerate(orders):
        restored[block].index_copy_(1, order.to(device=restored.device), decoded_permuted[block])
    return restored


def _batched_exact_blocks(
    values: torch.Tensor,
    orders: Sequence[torch.Tensor] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Solve B blocks at once and return [B,N,64] decoded coordinates."""

    # values is [N,B,64]; the solver can process independent blocks in one batch.
    if values.ndim != 3 or int(values.shape[-1]) != 64:
        raise ValueError(f"expected [N,B,64], got {tuple(values.shape)}")
    tokens, block_count = int(values.shape[0]), int(values.shape[1])
    if orders is None:
        ordered = values
        order_list = [torch.arange(64, device=values.device, dtype=torch.int64) for _ in range(block_count)]
    else:
        if len(orders) != block_count:
            raise ValueError("order count does not match block count")
        order_list = [order.to(device=values.device, dtype=torch.int64) for order in orders]
        ordered = torch.stack(
            [values[:, block].index_select(-1, order_list[block]) for block in range(block_count)],
            dim=1,
        )
    params, loss = exact_legal_blocks(ordered.reshape(tokens * block_count, 64))
    decoded = _decode_exact(params, tokens * block_count).reshape(tokens, block_count, 64).permute(1, 0, 2).contiguous()
    if orders is not None:
        decoded = _restore_permuted_blocks(decoded, order_list)
    return decoded, loss.reshape(tokens, block_count).transpose(0, 1).contiguous()


def _replace_block_values(
    base: torch.Tensor,
    rows: Sequence[int],
    block: int,
    values: torch.Tensor,
) -> torch.Tensor:
    result = base.clone()
    row_tensor = torch.as_tensor(rows, dtype=torch.int64, device=base.device)
    lo = int(block) * 64
    result[row_tensor, lo:lo + 64] = values
    return result


def _linear_partition_record(
    raw: Any,
    parent_path: Path,
    layer: int,
    role: str,
    args: argparse.Namespace,
    settings: Mapping[str, Any],
    prepared_cache: dict[int, tuple[Any, dict[tuple[int, str], tuple[Any, dict[str, torch.Tensor]]]]],
) -> tuple[dict[str, Any], str]:
    device = torch.device(args.device)
    state, _, artifact = _load_raw_and_state(
        raw, parent_path, layer, role, args.device, prepared_cache
    )
    weight_pair = v2._move_pair(v2._pair(raw.weights[layer][role]), device)
    weight_reference = v2.dequantize_nvfp4(*weight_pair).to(torch.float32)
    weight_continuous = _state_parent_transform(weight_reference, state)
    rows = _uniform_indices(int(weight_continuous.shape[0]), int(settings["weight_rows"]))
    block_count = int(weight_continuous.shape[1]) // 64
    blocks = _uniform_indices(block_count, int(settings["blocks"]))
    row_tensor = torch.as_tensor(rows, dtype=torch.int64, device=device)

    activation_parts: list[torch.Tensor] = []
    activation_rows_by_window: dict[int, list[int]] = {}
    for calibration_index in range(len(raw.calibration_windows)):
        pair = v2._move_pair(
            v2._pair(raw.calibration_activations[role][calibration_index][layer]), device
        )
        activation_reference = v2.dequantize_nvfp4(*pair).to(torch.float32)
        activation_continuous = _state_activation_transform(activation_reference, state)
        selected_rows = _uniform_indices(int(activation_continuous.shape[0]), int(settings["activation_tokens"]))
        activation_parts.append(activation_continuous[torch.as_tensor(selected_rows, dtype=torch.int64, device=device)])
        activation_rows_by_window[calibration_index] = list(selected_rows)
    activation_continuous = torch.cat(activation_parts, dim=0)
    weight_selected = weight_continuous[row_tensor]

    activation_values = torch.stack(
        [activation_continuous[:, block * 64:(block + 1) * 64] for block in blocks], dim=1
    )
    weight_values = torch.stack(
        [weight_selected[:, block * 64:(block + 1) * 64] for block in blocks], dim=1
    )
    orders: list[torch.Tensor] = []
    pressure_records: list[dict[str, Any]] = []
    for block_position, block in enumerate(blocks):
        order, activation_rms, weight_rms = _pressure_order(
            activation_values[:, block_position], weight_values[:, block_position]
        )
        orders.append(order)
        pressure_records.append({
            "block": int(block),
            "order": [int(value) for value in order.detach().cpu().tolist()],
            "pressure_p50_p90_p99": [float(value) for value in torch.quantile(
                torch.maximum(
                    torch.log2((activation_rms / activation_rms.median().clamp_min(EPS)).clamp_min(EPS)),
                    torch.log2((weight_rms / weight_rms.median().clamp_min(EPS)).clamp_min(EPS)),
                ),
                torch.tensor((0.50, 0.90, 0.99), device=device),
            ).detach().cpu().tolist()],
        })

    fixed_activation, fixed_activation_loss = _batched_exact_blocks(activation_values)
    candidate_activation, candidate_activation_loss = _batched_exact_blocks(activation_values, orders)
    fixed_weight, fixed_weight_loss = _batched_exact_blocks(weight_values)
    candidate_weight, candidate_weight_loss = _batched_exact_blocks(weight_values, orders)

    activation_fixed = activation_continuous.clone()
    activation_candidate = activation_continuous.clone()
    weight_fixed = weight_continuous.clone()
    weight_candidate = weight_continuous.clone()
    for position, block in enumerate(blocks):
        activation_fixed[:, block * 64:(block + 1) * 64] = fixed_activation[position]
        activation_candidate[:, block * 64:(block + 1) * 64] = candidate_activation[position]
        weight_fixed[row_tensor, block * 64:(block + 1) * 64] = fixed_weight[position]
        weight_candidate[row_tensor, block * 64:(block + 1) * 64] = candidate_weight[position]
    reference_output = activation_continuous @ weight_continuous.T
    fixed_output = activation_fixed @ weight_fixed.T
    candidate_output = activation_candidate @ weight_candidate.T
    fixed_mse = float((fixed_output - reference_output).square().mean())
    candidate_mse = float((candidate_output - reference_output).square().mean())
    output_gain = (fixed_mse - candidate_mse) / max(fixed_mse, EPS)
    fixed_operand_sse = float(
        (fixed_activation - activation_values.permute(1, 0, 2)).square().sum()
        + (fixed_weight - weight_values.permute(1, 0, 2)).square().sum()
    )
    candidate_operand_sse = float(
        (candidate_activation - activation_values.permute(1, 0, 2)).square().sum()
        + (candidate_weight - weight_values.permute(1, 0, 2)).square().sum()
    )
    record = {
        "layer": layer,
        "role": role,
        "shape": list(weight_reference.shape),
        "rows": list(rows),
        "blocks": list(blocks),
        "calibration_windows": list(range(len(raw.calibration_windows))),
        "activation_rows_by_window": activation_rows_by_window,
        "parent_calibration_artifact": artifact,
        "pressure": pressure_records,
        "fixed_partition": {
            "output_mse": fixed_mse,
            "operand_sse": fixed_operand_sse,
            "solver_loss_activation": float(fixed_activation_loss.sum()),
            "solver_loss_weight": float(fixed_weight_loss.sum()),
        },
        "shared_pressure_partition": {
            "output_mse": candidate_mse,
            "operand_sse": candidate_operand_sse,
            "solver_loss_activation": float(candidate_activation_loss.sum()),
            "solver_loss_weight": float(candidate_weight_loss.sum()),
            "output_gain_vs_fixed": output_gain,
        },
        "reference_output_mse": 0.0,
        "finite": True,
    }
    return record, artifact


def _attention_metrics(
    q: torch.Tensor,
    k: torch.Tensor,
    value: torch.Tensor,
    raw: Any,
    reference_output: torch.Tensor,
    reference_logits: torch.Tensor,
    reference_probabilities: torch.Tensor,
) -> dict[str, float]:
    output, logits, probabilities = v2._attention_trace(
        q[None], k[None], value[None], raw.q_heads, raw.kv_heads, raw.head_dim
    )
    return {
        "output_mse": float((output - reference_output).square().mean()),
        "logit_mse": float((logits - reference_logits).square().mean()),
        "probability_mse": float((probabilities - reference_probabilities).square().mean()),
    }


def _attention_partition_record(
    raw: Any,
    parent_path: Path,
    layer: int,
    args: argparse.Namespace,
    settings: Mapping[str, Any],
    state_cache: dict[int, tuple[Any, dict[int, dict[str, Any]]]],
) -> tuple[dict[str, Any], str]:
    device = torch.device(args.device)
    states, artifact = _load_parent_attention_state(raw, parent_path, layer, args.device, state_cache)
    q_head = int(settings["q_head"])
    kv_head = raw.kv_heads - 1 if str(settings["kv_head"]) == "last" else int(settings["kv_head"])
    token_count = int(settings["tokens"])
    q_lo = q_head * raw.head_dim
    k_lo = kv_head * raw.head_dim
    records: list[dict[str, Any]] = []

    for calibration_index, item in enumerate(raw.calibration_qkv):
        q_raw, k_raw, v_raw = item[layer]
        q_pair = v2._move_pair(v2._pair(q_raw), device)
        k_pair = v2._move_pair(v2._pair(k_raw), device)
        v_pair = v2._move_pair(v2._pair(v_raw), device)
        q_reference = v2.dequantize_nvfp4(*q_pair).to(torch.float32)
        k_reference = v2.dequantize_nvfp4(*k_pair).to(torch.float32)
        v_reference = v2.dequantize_nvfp4(*v_pair).to(torch.float32)
        q_params = sol.hif4_dynamic_quantize_q(
            q_pair[0], q_pair[1], raw.q_heads, raw.head_dim, states["q_state"]
        )
        k_params = sol.hif4_dynamic_quantize_k(
            k_pair[0], k_pair[1], raw.kv_heads, raw.head_dim, states["k_state"]
        )
        v_params = sol.hif4_dynamic_quantize_v(
            v_pair[0], v_pair[1], raw.kv_heads, raw.head_dim, states["v_state"]
        )
        q_parent = ref.dequantize_hif4(q_params, tuple(q_reference.shape)).to(torch.float32)
        k_parent = ref.dequantize_hif4(k_params, tuple(k_reference.shape)).to(torch.float32)
        v_parent = ref.dequantize_hif4(v_params, tuple(v_reference.shape)).to(torch.float32)
        q_continuous = sol._attention_state_transform_dense(
            q_reference, states["q_state"], raw.q_heads, raw.head_dim, is_k=False
        )
        k_continuous = sol._attention_state_transform_dense(
            k_reference, states["k_state"], raw.kv_heads, raw.head_dim, is_k=True
        )
        q_rows = _uniform_indices(int(q_reference.shape[0]), token_count)
        k_rows = _uniform_indices(int(k_reference.shape[0]), token_count)
        q_row_tensor = torch.as_tensor(q_rows, dtype=torch.int64, device=device)
        k_row_tensor = torch.as_tensor(k_rows, dtype=torch.int64, device=device)
        q_values = q_continuous[q_row_tensor, q_lo:q_lo + raw.head_dim]
        k_values = k_continuous[k_row_tensor, k_lo:k_lo + raw.head_dim]
        order, q_rms, k_rms = _pressure_order(q_values, k_values)
        q_fixed_params, q_fixed_loss = exact_legal_blocks(q_values)
        q_candidate_params, q_candidate_loss = exact_legal_blocks(q_values.index_select(-1, order))
        k_fixed_params, k_fixed_loss = exact_legal_blocks(k_values)
        k_candidate_params, k_candidate_loss = exact_legal_blocks(k_values.index_select(-1, order))
        q_fixed = ref.dequantize_hif4(q_fixed_params, (len(q_rows), raw.head_dim)).to(torch.float32)
        q_candidate_permuted = ref.dequantize_hif4(q_candidate_params, (len(q_rows), raw.head_dim)).to(torch.float32)
        q_candidate = torch.empty_like(q_candidate_permuted)
        q_candidate[:, order] = q_candidate_permuted
        k_fixed = ref.dequantize_hif4(k_fixed_params, (len(k_rows), raw.head_dim)).to(torch.float32)
        k_candidate_permuted = ref.dequantize_hif4(k_candidate_params, (len(k_rows), raw.head_dim)).to(torch.float32)
        k_candidate = torch.empty_like(k_candidate_permuted)
        k_candidate[:, order] = k_candidate_permuted

        q_fixed_full = q_parent.clone()
        q_candidate_full = q_parent.clone()
        k_fixed_full = k_parent.clone()
        k_candidate_full = k_parent.clone()
        q_fixed_full[q_row_tensor, q_lo:q_lo + raw.head_dim] = q_fixed
        q_candidate_full[q_row_tensor, q_lo:q_lo + raw.head_dim] = q_candidate
        k_fixed_full[k_row_tensor, k_lo:k_lo + raw.head_dim] = k_fixed
        k_candidate_full[k_row_tensor, k_lo:k_lo + raw.head_dim] = k_candidate

        reference_output, reference_logits, reference_probabilities = v2._attention_trace(
            q_reference[None], k_reference[None], v_reference[None],
            raw.q_heads, raw.kv_heads, raw.head_dim,
        )
        parent_metrics = _attention_metrics(
            q_parent, k_parent, v_parent, raw,
            reference_output, reference_logits, reference_probabilities,
        )
        fixed_q_metrics = _attention_metrics(
            q_fixed_full, k_parent, v_parent, raw,
            reference_output, reference_logits, reference_probabilities,
        )
        candidate_q_metrics = _attention_metrics(
            q_candidate_full, k_parent, v_parent, raw,
            reference_output, reference_logits, reference_probabilities,
        )
        fixed_k_metrics = _attention_metrics(
            q_parent, k_fixed_full, v_parent, raw,
            reference_output, reference_logits, reference_probabilities,
        )
        candidate_k_metrics = _attention_metrics(
            q_parent, k_candidate_full, v_parent, raw,
            reference_output, reference_logits, reference_probabilities,
        )
        fixed_joint_metrics = _attention_metrics(
            q_fixed_full, k_fixed_full, v_parent, raw,
            reference_output, reference_logits, reference_probabilities,
        )
        candidate_joint_metrics = _attention_metrics(
            q_candidate_full, k_candidate_full, v_parent, raw,
            reference_output, reference_logits, reference_probabilities,
        )
        q_gain = (fixed_q_metrics["output_mse"] - candidate_q_metrics["output_mse"]) / max(fixed_q_metrics["output_mse"], EPS)
        k_gain = (fixed_k_metrics["output_mse"] - candidate_k_metrics["output_mse"]) / max(fixed_k_metrics["output_mse"], EPS)
        joint_gain = (fixed_joint_metrics["output_mse"] - candidate_joint_metrics["output_mse"]) / max(fixed_joint_metrics["output_mse"], EPS)
        record = {
            "layer": layer,
            "calibration_window": calibration_index,
            "length": int(q_reference.shape[0]),
            "q_rows": list(q_rows),
            "k_rows": list(k_rows),
            "q_head": q_head,
            "kv_head": kv_head,
            "shared_pressure_order": [int(value) for value in order.detach().cpu().tolist()],
            "pressure": {
                "q_rms_p50_p90_p99": [float(value) for value in torch.quantile(q_rms, torch.tensor((0.50, 0.90, 0.99), device=device)).detach().cpu().tolist()],
                "k_rms_p50_p90_p99": [float(value) for value in torch.quantile(k_rms, torch.tensor((0.50, 0.90, 0.99), device=device)).detach().cpu().tolist()],
            },
            "parent": parent_metrics,
            "q_fixed": {**fixed_q_metrics, "solver_loss": float(q_fixed_loss.sum())},
            "q_oracle": {**candidate_q_metrics, "output_gain_vs_fixed": q_gain, "solver_loss": float(q_candidate_loss.sum())},
            "k_fixed": {**fixed_k_metrics, "solver_loss": float(k_fixed_loss.sum())},
            "k_oracle": {**candidate_k_metrics, "output_gain_vs_fixed": k_gain, "solver_loss": float(k_candidate_loss.sum())},
            "joint_fixed": fixed_joint_metrics,
            "joint_oracle": {**candidate_joint_metrics, "output_gain_vs_fixed": joint_gain},
            "parent_calibration_artifact": artifact,
            "finite": True,
        }
        records.append(record)

    if len(raw.calibration_qkv) == 0:
        raise ValueError("attention cache has no calibration windows")
    return {"records": records}, artifact


def _gain_list(records: Sequence[Mapping[str, Any]], arm: str) -> list[float]:
    return [float(item[arm]["output_gain_vs_fixed"]) for item in records]


def run_d_b(
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
                record, artifact = _linear_partition_record(
                    raw, parent_path, layer, role, args, linear_settings, prepared_cache
                )
                linear_records.append(record)
                artifact_paths.add(artifact)

    attention_state_cache: dict[int, tuple[Any, dict[int, dict[str, Any]]]] = {}
    attention_records: list[dict[str, Any]] = []
    with torch.inference_mode():
        for layer in layers:
            packed, artifact = _attention_partition_record(
                raw, parent_path, layer, args, attention_settings, attention_state_cache
            )
            attention_records.extend(packed["records"])
            artifact_paths.add(artifact)

    layer_summary: list[dict[str, Any]] = []
    for layer in layers:
        focus = [item for item in linear_records if item["layer"] == layer and item["role"] in focus_roles]
        all_roles = [item for item in linear_records if item["layer"] == layer]
        focus_gains = torch.tensor([item["shared_pressure_partition"]["output_gain_vs_fixed"] for item in focus], dtype=torch.float64)
        all_gains = torch.tensor([item["shared_pressure_partition"]["output_gain_vs_fixed"] for item in all_roles], dtype=torch.float64)
        layer_summary.append({
            "layer": layer,
            "focus_median_output_gain": float(torch.median(focus_gains)) if focus else 0.0,
            "focus_positive_roles": int((focus_gains > 0.0).sum()) if focus else 0,
            "focus_role_count": len(focus),
            "all_role_median_output_gain": float(torch.median(all_gains)) if all_roles else 0.0,
        })
    gate_settings = config["gate"]
    threshold = float(gate_settings["focus_layer_median_gain"])
    positive_layers = [
        item["layer"] for item in layer_summary if item["focus_median_output_gain"] > threshold
    ]
    deep_positive = any(int(layer) >= 15 for layer in positive_layers)
    non_deep_positive = any(int(layer) < 15 for layer in positive_layers)
    gate = {
        "focus_layer_median_gain": {
            str(item["layer"]): item["focus_median_output_gain"] for item in layer_summary
        },
        "positive_focus_layers": len(positive_layers),
        "positive_focus_layer_ids": positive_layers,
        "layer_count": len(layers),
        "deep_positive": deep_positive,
        "non_deep_positive": non_deep_positive,
        "threshold": threshold,
    }
    gate_pass = (
        len(positive_layers) >= int(gate_settings["positive_focus_layers"])
        and deep_positive
        and non_deep_positive
    )
    q_gains = _gain_list(attention_records, "q_oracle")
    k_gains = _gain_list(attention_records, "k_oracle")
    joint_gains = _gain_list(attention_records, "joint_oracle")
    attention_summary = {
        "records": len(attention_records),
        "q_median_gain": float(torch.median(torch.tensor(q_gains, dtype=torch.float64))) if q_gains else 0.0,
        "k_median_gain": float(torch.median(torch.tensor(k_gains, dtype=torch.float64))) if k_gains else 0.0,
        "joint_median_gain": float(torch.median(torch.tensor(joint_gains, dtype=torch.float64))) if joint_gains else 0.0,
        "q_mean_gain": sum(q_gains) / max(len(q_gains), 1),
        "k_mean_gain": sum(k_gains) / max(len(k_gains), 1),
        "joint_mean_gain": sum(joint_gains) / max(len(joint_gains), 1),
    }
    status = "PASS_TO_R3" if gate_pass else "REJECTED"
    expected = {
        "layers": layers,
        "linear_records": len(layers) * len(roles),
        "attention_records": len(layers) * len(raw.calibration_windows),
        "linear_blocks_per_record": int(linear_settings["blocks"]),
        "objective": "legal fixed partition versus fixed shared-pressure block-local permutation",
    }
    executed = {
        "linear_records": len(linear_records),
        "attention_records": len(attention_records),
        "linear_exact_solver_calls": len(linear_records) * 4,
        "attention_exact_solver_arms": len(attention_records) * 4,
        "parent_artifacts": sorted(artifact_paths),
        "elapsed_s": time.perf_counter() - started,
    }
    result = {
        "stage": "d-b",
        "status": status,
        "hypothesis": "Reassigning values to 8x8x4 subgroups inside each legal 64-block can lower real output error without changing the five fields.",
        "expected_cases": expected,
        "executed_cases": executed,
        "failed_cases": [],
        "metrics": {
            "gate": gate,
            "linear_layer_summary": layer_summary,
            "linear_records": linear_records,
            "attention_summary": attention_summary,
            "attention_records": attention_records,
            "elapsed_s": time.perf_counter() - started,
        },
        "gate": "D_B_PASS_TO_R3" if gate_pass else "NO_MATERIAL_GROUPING_ORACLE",
        "reason": (
            "The fixed D-B gate passed; the next allowed step is a separate plan for one deployable shared-pressure rule."
            if gate_pass else
            "The fixed D-B gate did not show >1% focus output gain on at least three layers with both depth bands; no grouping candidate is allowed."
        ),
        "next_step": "r3-plan" if gate_pass else "close-plan",
    }
    manifest = _manifest("d-b", args, config_path, cache_path, parent_path, expected, executed, result["status"])
    _write_stage(output_dir, result, manifest)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("d-a", "d-b"), required=True)
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
    if args.stage == "d-a":
        return run_d_a(args, config, config_path, output_dir, parent_path, cache_path)
    return run_d_b(args, config, config_path, output_dir, parent_path, cache_path)


if __name__ == "__main__":
    result = main()
    print(json.dumps({"stage": result["stage"], "status": result["status"], "next_step": result["next_step"]}, ensure_ascii=False))
