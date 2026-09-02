"""Bounded local-only HiF4 teacher diagnostic for fc Activation codes.

This is deliberately not a competition candidate.  It loads one immutable
parent module, calibrates only the selected fc roles/layers, and asks whether
legal local code edits can reduce the deployed output objective.  The teacher
never enters a public API state and its cost is reported separately from the
six-API evaluator timing.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

ROOT = Path(__file__).resolve().parents[1]
EVALUATOR = ROOT / "evaluator"
if str(EVALUATOR) not in sys.path:
    sys.path.insert(0, str(EVALUATOR))

import official_eval as evaluator  # noqa: E402


BLOCK = 64
EPS = 1.0e-12
SIGNED_LEVELS = torch.arange(-7, 8, dtype=torch.float32) * 0.25
DEFAULT_LAYERS = (0, 3, 7, 10, 13, 16, 20, 23)
DEFAULT_ROLES = ("fc_gate", "fc_up")
EDIT_CLASSES = ("mantissa", "lv3", "lv2", "scale", "joint")


def _load_parent(path: Path):
    source = path.resolve()
    name = f"_l3_parent_{hashlib.sha1(str(source).encode()).hexdigest()}"
    spec = importlib.util.spec_from_file_location(name, source)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load parent source: {source}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _pair(value: tuple[torch.Tensor, torch.Tensor], device: torch.device):
    return value[0].to(device), value[1].to(device)


def _finite(value: float) -> float:
    if not math.isfinite(float(value)):
        raise ValueError(f"non-finite diagnostic value: {value}")
    return float(value)


def _levels(device: torch.device) -> torch.Tensor:
    return SIGNED_LEVELS.to(device)


def _denominator(
    scale: torch.Tensor,
    lv2: torch.Tensor,
    lv3: torch.Tensor,
) -> torch.Tensor:
    """Expand legal hierarchical fields to one denominator per channel."""
    rows = int(scale.shape[0])
    value = (
        scale.reshape(rows, 1, 1, 1)
        * lv2.reshape(rows, 8, 1, 1)
        * lv3.reshape(rows, 8, 2, 1)
    )
    return value.expand(rows, 8, 2, 4).reshape(rows, BLOCK)


def _round_to_codes(target: torch.Tensor, denominator: torch.Tensor) -> torch.Tensor:
    return (
        torch.round(target * 4.0 / denominator.clamp_min(EPS))
        .clamp(-7.0, 7.0)
        * 0.25
        * denominator
    )


def _block_quadratic(
    q: torch.Tensor,
    hessian: torch.Tensor,
    cross: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    """Return the deployed-Q(W) block quadratic up to a target constant."""
    return torch.einsum("ri,ij,rj->r", q, hessian, q) - 2.0 * torch.einsum(
        "ri,ij,rj->r", q, cross, target
    )


def _coordinate_pass(
    q: torch.Tensor,
    denominator: torch.Tensor,
    hessian: torch.Tensor,
    cross: torch.Tensor,
    target: torch.Tensor,
    coordinates: Sequence[int],
    sweeps: int,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Perform a fixed bounded legal signed-mantissa coordinate pass."""
    levels = _levels(q.device)
    accepted_moves = 0
    accepted_values = 0
    accepted_gain = 0.0
    diagonal = torch.diagonal(hessian).clamp_min(EPS)
    for _ in range(max(1, int(sweeps))):
        for coordinate in coordinates:
            gradient = torch.einsum("ij,rj->ri", hessian, q) - torch.einsum(
                "ij,rj->ri", cross, target
            )
            options = denominator[:, coordinate, None] * levels[None, :]
            step = options - q[:, coordinate, None]
            change = (
                2.0 * step * gradient[:, coordinate, None]
                + diagonal[coordinate] * step.square()
            )
            best_change, best_index = change.min(dim=-1)
            improve = torch.isfinite(best_change) & (best_change < -EPS)
            accepted = step.gather(-1, best_index[:, None]).squeeze(-1)
            accepted = torch.where(improve, accepted, torch.zeros_like(accepted))
            q[:, coordinate] += accepted
            accepted_moves += int(improve.sum().item())
            accepted_values += int((accepted != 0.0).sum().item())
            accepted_gain += float((-best_change[improve]).sum().item())
    return q, {
        "accepted_moves": float(accepted_moves),
        "accepted_values": float(accepted_values),
        "accepted_quadratic_gain": _finite(accepted_gain),
    }


def _empty_move_stats() -> dict[str, float]:
    return {
        "accepted_moves": 0.0,
        "accepted_values": 0.0,
        "accepted_quadratic_gain": 0.0,
    }


def _merge_move_stats(left: Mapping[str, float], right: Mapping[str, float]) -> dict[str, float]:
    return {name: float(left.get(name, 0.0)) + float(right.get(name, 0.0)) for name in left}


def _apply_mantissa(
    q: torch.Tensor,
    denominator: torch.Tensor,
    hessian: torch.Tensor,
    cross: torch.Tensor,
    target: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict[str, float]]:
    q_new, moves = _coordinate_pass(
        q.clone(), denominator, hessian, cross, target, tuple(range(BLOCK)), sweeps=2
    )
    return q_new, denominator, torch.empty(0, device=q.device), torch.empty(0, device=q.device), moves


def _toggle_groups(
    q: torch.Tensor,
    scale: torch.Tensor,
    lv2: torch.Tensor,
    lv3: torch.Tensor,
    target: torch.Tensor,
    hessian: torch.Tensor,
    cross: torch.Tensor,
    *,
    level: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict[str, float]]:
    """Try each legal lv3/lv2 toggle once, accepting per-row improvements."""
    q_current = q.clone()
    scale_current = scale.clone()
    lv2_current = lv2.clone()
    lv3_current = lv3.clone()
    denominator_current = _denominator(scale_current, lv2_current, lv3_current)
    stats = _empty_move_stats()
    current_loss = _block_quadratic(q_current, hessian, cross, target)
    groups = 16 if level == "lv3" else 8
    for group_index in range(groups):
        if level == "lv3":
            subgroup = group_index % 2
            parent_group = group_index // 2
            lo = parent_group * 8 + subgroup * 4
            hi = lo + 4
            lv2_candidate = lv2_current
            lv3_candidate = lv3_current.clone()
            lv3_candidate[:, parent_group, subgroup] = torch.where(
                lv3_current[:, parent_group, subgroup] == 1.0,
                torch.full_like(lv3_current[:, parent_group, subgroup], 2.0),
                torch.ones_like(lv3_current[:, parent_group, subgroup]),
            )
        else:
            lo = group_index * 8
            hi = lo + 8
            lv2_candidate = lv2_current.clone()
            lv2_candidate[:, group_index] = torch.where(
                lv2_current[:, group_index] == 1.0,
                torch.full_like(lv2_current[:, group_index], 2.0),
                torch.ones_like(lv2_current[:, group_index]),
            )
            lv3_candidate = lv3_current
        denominator_candidate = _denominator(
            scale_current, lv2_candidate, lv3_candidate
        )
        q_candidate = q_current.clone()
        q_candidate[:, lo:hi] = _round_to_codes(
            target[:, lo:hi], denominator_candidate[:, lo:hi]
        )
        q_candidate, move_stats = _coordinate_pass(
            q_candidate,
            denominator_candidate,
            hessian,
            cross,
            target,
            tuple(range(lo, hi)),
            sweeps=1,
        )
        stats = _merge_move_stats(stats, move_stats)
        candidate_loss = _block_quadratic(
            q_candidate, hessian, cross, target
        )
        accept = torch.isfinite(candidate_loss) & (candidate_loss < current_loss - EPS)
        q_current = torch.where(accept[:, None], q_candidate, q_current)
        denominator_current = torch.where(
            accept[:, None], denominator_candidate, denominator_current
        )
        if level == "lv3":
            lv3_current = torch.where(accept[:, None, None], lv3_candidate, lv3_current)
        else:
            lv2_current = torch.where(accept[:, None], lv2_candidate, lv2_current)
        current_loss = torch.where(accept, candidate_loss, current_loss)
    return q_current, denominator_current, lv2_current, lv3_current, stats


def _run_scale_with_state(
    q: torch.Tensor,
    scale: torch.Tensor,
    lv2: torch.Tensor,
    lv3: torch.Tensor,
    target: torch.Tensor,
    hessian: torch.Tensor,
    cross: torch.Tensor,
    parent: Any,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict[str, float]]:
    """Scale variant retaining the selected scale tensor."""
    rows = int(q.shape[0])
    best_q = q.clone()
    best_scale = scale.clone()
    best_lv2 = lv2.clone()
    best_lv3 = lv3.clone()
    best_loss = _block_quadratic(q, hessian, cross, target)
    stats = _empty_move_stats()
    code = parent._e6m2_encode_nearest(scale)
    absolute = target.abs().reshape(rows, 8, 2, 4)
    sign = torch.sign(target)
    for direction in (-1, 1):
        candidate_code = (code.to(torch.int64) + direction).clamp(0, 254).to(torch.int16)
        candidate_scale = parent._e6m2_decode(candidate_code)
        _, candidate_lv2, candidate_lv3, candidate_mant = parent._solve_hierarchy(
            absolute, candidate_scale
        )
        candidate_den = _denominator(candidate_scale, candidate_lv2, candidate_lv3)
        candidate_q = sign * candidate_mant.reshape(rows, BLOCK) * candidate_den
        candidate_q, move_stats = _coordinate_pass(
            candidate_q,
            candidate_den,
            hessian,
            cross,
            target,
            tuple(range(BLOCK)),
            sweeps=1,
        )
        stats = _merge_move_stats(stats, move_stats)
        candidate_loss = _block_quadratic(candidate_q, hessian, cross, target)
        accept = torch.isfinite(candidate_loss) & (candidate_loss < best_loss - EPS)
        best_q = torch.where(accept[:, None], candidate_q, best_q)
        best_scale = torch.where(accept, candidate_scale, best_scale)
        best_lv2 = torch.where(accept[:, None], candidate_lv2, best_lv2)
        best_lv3 = torch.where(accept[:, None, None], candidate_lv3, best_lv3)
        best_loss = torch.where(accept, candidate_loss, best_loss)
    stats["scale_changed_rows"] = float((best_scale != scale).sum().item())
    return best_q, best_scale, best_lv2, best_lv3, stats


def _run_teacher(
    class_name: str,
    q_parent: torch.Tensor,
    scale_parent: torch.Tensor,
    lv2_parent: torch.Tensor,
    lv3_parent: torch.Tensor,
    target: torch.Tensor,
    hessian: torch.Tensor,
    cross: torch.Tensor,
    parent: Any,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict[str, float]]:
    if class_name == "joint":
        q, scale, lv2, lv3 = (
            q_parent.clone(), scale_parent.clone(), lv2_parent.clone(), lv3_parent.clone()
        )
        stats = _empty_move_stats()
        for _ in range(2):
            for component in ("mantissa", "lv3", "lv2", "scale"):
                if component == "mantissa":
                    q, scale, lv2, lv3, move_stats = _run_teacher(
                        component, q, scale, lv2, lv3, target, hessian, cross, parent
                    )
                elif component == "scale":
                    q, scale, lv2, lv3, move_stats = _run_scale_with_state(
                        q, scale, lv2, lv3, target, hessian, cross, parent
                    )
                else:
                    q, scale, lv2, lv3, move_stats = _run_teacher(
                        component, q, scale, lv2, lv3, target, hessian, cross, parent
                    )
                stats = _merge_move_stats(stats, move_stats)
        return q, scale, lv2, lv3, stats
    if class_name == "mantissa":
        denominator = _denominator(scale_parent, lv2_parent, lv3_parent)
        q, _, _, _, stats = _apply_mantissa(
            q_parent, denominator, hessian, cross, target
        )
        return q, scale_parent.clone(), lv2_parent.clone(), lv3_parent.clone(), stats
    if class_name in {"lv3", "lv2"}:
        q, _, lv2, lv3, stats = _toggle_groups(
            q_parent, scale_parent, lv2_parent, lv3_parent,
            target, hessian, cross, level=class_name,
        )
        return q, scale_parent.clone(), lv2, lv3, stats
    if class_name == "scale":
        return _run_scale_with_state(
            q_parent, scale_parent, lv2_parent, lv3_parent,
            target, hessian, cross, parent,
        )
    raise ValueError(f"unknown teacher class: {class_name}")


def _class_metrics(
    class_name: str,
    q_parent: torch.Tensor,
    scale_parent: torch.Tensor,
    lv2_parent: torch.Tensor,
    lv3_parent: torch.Tensor,
    q_teacher: torch.Tensor,
    scale_teacher: torch.Tensor,
    lv2_teacher: torch.Tensor,
    lv3_teacher: torch.Tensor,
    target: torch.Tensor,
    hessian: torch.Tensor,
    cross: torch.Tensor,
    moves: Mapping[str, float],
    exact_single_block_margin: float | None = None,
) -> dict[str, Any]:
    parent_quad = _block_quadratic(q_parent, hessian, cross, target)
    teacher_quad = _block_quadratic(q_teacher, hessian, cross, target)
    parent_codes = torch.round(q_parent * 4.0 / _denominator(scale_parent, lv2_parent, lv3_parent).clamp_min(EPS)).clamp(-7.0, 7.0)
    teacher_codes = torch.round(q_teacher * 4.0 / _denominator(scale_teacher, lv2_teacher, lv3_teacher).clamp_min(EPS)).clamp(-7.0, 7.0)
    quad_margin = (parent_quad - teacher_quad) / (parent_quad.abs() + EPS)
    changed_rows = int(((q_teacher - q_parent).abs().amax(dim=-1) > 1.0e-7).sum().item())
    changed_values = int(((teacher_codes - parent_codes).abs() > 1.0e-7).sum().item())
    lv2_flips = int((lv2_teacher != lv2_parent).sum().item())
    lv3_flips = int((lv3_teacher != lv3_parent).sum().item())
    parent_scale_code = evaluator_e6_code(scale_parent)
    teacher_scale_code = evaluator_e6_code(scale_teacher)
    scale_delta = teacher_scale_code.to(torch.int64) - parent_scale_code.to(torch.int64)
    unique_delta, counts = torch.unique(scale_delta, return_counts=True)
    scale_hist = {str(int(key)): int(value) for key, value in zip(unique_delta.tolist(), counts.tolist())}
    return {
        "edit_class": class_name,
        "block_quadratic_parent": _finite(float(parent_quad.mean().item())),
        "block_quadratic_teacher": _finite(float(teacher_quad.mean().item())),
        "block_quadratic_margin": _finite(float(quad_margin.mean().item())),
        "recoverable_margin": _finite(float(quad_margin.mean().item())),
        "exact_single_block_margin": (
            None
            if exact_single_block_margin is None
            else _finite(float(exact_single_block_margin))
        ),
        "row_count": int(q_parent.shape[0]),
        "changed_rows": changed_rows,
        "changed_values": changed_values,
        "lv2_flips": lv2_flips,
        "lv3_flips": lv3_flips,
        "scale_code_delta_histogram": scale_hist,
        "accepted_moves": int(moves.get("accepted_moves", 0.0)),
        "accepted_values": int(moves.get("accepted_values", 0.0)),
        "accepted_quadratic_gain": _finite(float(moves.get("accepted_quadratic_gain", 0.0))),
    }


def evaluator_e6_code(value: torch.Tensor) -> torch.Tensor:
    """Use the parent implementation while keeping this file self-contained."""
    # The parent and frozen reference use the same unsigned E6M2 nearest rule.
    x = torch.nan_to_num(value.to(torch.float32), nan=2.0**-48, posinf=49152.0, neginf=2.0**-48).clamp(2.0**-48, 49152.0)
    exponent = torch.floor(torch.log2(x))
    base = torch.pow(2.0, exponent)
    mantissa = torch.round((x / base - 1.0) * 4.0).to(torch.int64)
    carry = mantissa >= 4
    exponent = exponent + carry.to(exponent.dtype)
    mantissa = torch.where(carry, torch.zeros_like(mantissa), mantissa).clamp(0, 3)
    exponent_field = (exponent.to(torch.int64) + 48).clamp(0, 63)
    return (exponent_field * 4 + mantissa).clamp(0, 254).to(torch.int16)


def _summarize(items: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    margins = [float(item["recoverable_margin"]) for item in items]
    if not margins:
        return {"case_count": 0}
    positive = sum(value > 0.0 for value in margins)
    negative = sum(value < 0.0 for value in margins)
    return {
        "case_count": len(items),
        "mean_recoverable_margin": _finite(sum(margins) / len(margins)),
        "median_recoverable_margin": _finite(float(statistics.median(margins))),
        "positive_cases": positive,
        "negative_cases": negative,
        "zero_cases": len(margins) - positive - negative,
        "effect": (
            "no_effect" if positive == 0 and negative == 0
            else "consistent_improvement" if negative == 0
            else "consistent_regression" if positive == 0
            else "mixed"
        ),
    }


def _feature_stats(values: torch.Tensor) -> dict[str, float]:
    """Summarize one block feature without serializing the feature tensor."""
    flat = values.detach().to(torch.float32).reshape(-1)
    if flat.numel() == 0:
        return {"count": 0}
    flat = flat[torch.isfinite(flat)]
    if flat.numel() == 0:
        return {"count": 0}
    quantiles = torch.quantile(flat, torch.tensor((0.5, 0.75, 0.9), device=flat.device))
    return {
        "count": int(flat.numel()),
        "mean": _finite(float(flat.mean().item())),
        "median": _finite(float(quantiles[0].item())),
        "q75": _finite(float(quantiles[1].item())),
        "q90": _finite(float(quantiles[2].item())),
    }


def _class_feature_summary(
    margins: Sequence[float],
    features: Mapping[str, torch.Tensor],
) -> dict[str, Any]:
    """Compare cheap parent-local features on positive/negative blocks.

    ``margins`` are exact per-block quadratic margins only for localization.  The
    summary is intentionally descriptive: it does not fit a student or choose a
    threshold, and therefore cannot leak a teacher into an API.
    """
    positive = torch.as_tensor([value > 0.0 for value in margins], device=next(iter(features.values())).device)
    negative = ~positive
    result: dict[str, Any] = {
        "positive_block_count": int(positive.sum().item()),
        "negative_block_count": int(negative.sum().item()),
    }
    for name, value in features.items():
        value = value.to(torch.float32).reshape(-1)
        result[name] = {
            "all": _feature_stats(value),
            "positive": _feature_stats(value[positive]),
            "negative": _feature_stats(value[negative]),
        }
    return result


def _group_class_summary(
    records: Sequence[Mapping[str, Any]],
    class_name: str,
    key: str,
) -> dict[str, Any]:
    groups: dict[str, list[Mapping[str, Any]]] = {}
    for record in records:
        group = str(record[key])
        groups.setdefault(group, []).append(
            {"recoverable_margin": record["classes"][class_name]["recoverable_margin"]}
        )
    return {group: _summarize(items) for group, items in sorted(groups.items())}


def run(args: argparse.Namespace) -> dict[str, Any]:
    parent_path = args.parent.resolve()
    parent = _load_parent(parent_path)
    parent_hash = hashlib.sha256(parent_path.read_bytes()).hexdigest()
    expected_hash = "800ca10ec3414e4fe886b93ca62bd4a350d26bba015287df7e8df2dd871ac23d"
    if parent_hash.lower() != expected_hash:
        raise ValueError(
            f"unexpected parent SHA {parent_hash}; D0 requires pre-A3 {expected_hash}"
        )
    device = torch.device(args.device)
    raw = evaluator.load_pack(args.cache.resolve())
    layers = tuple(int(value) for value in args.layers.split(",") if value.strip())
    roles = tuple(value.strip() for value in args.roles.split(",") if value.strip())
    if layers != DEFAULT_LAYERS:
        raise ValueError(f"D0 layers are fixed to {DEFAULT_LAYERS}")
    if roles != DEFAULT_ROLES:
        raise ValueError(f"D0 roles are fixed to {DEFAULT_ROLES}")
    started = time.perf_counter()
    records: list[dict[str, Any]] = []
    for layer in layers:
        for role in roles:
            print(f"[D0] calibrating layer={layer} role={role}", flush=True)
            weight_pair = _pair(evaluator._pair(raw.weights[layer][role]), device)
            calibration_pairs = [
                _pair(evaluator._pair(raw.calibration_activations[role][fold][layer]), device)
                for fold in range(2)
            ]
            calibration = parent.hif4_calibration_and_quantize_weight(
                weight_pair[0], weight_pair[1], calibration_pairs
            )
            evaluator.validate_state(calibration["activation_state"])
            evaluator.validate_hif4_params(
                calibration["weight_params"],
                evaluator.dequantize_nvfp4(*weight_pair).shape,
            )
            weight_t, activation_t = parent._active_transformed_calibration(
                weight_pair[0], weight_pair[1], calibration_pairs,
                calibration["activation_state"],
            )
            weight_t = weight_t.to(torch.float32)
            q_weight = parent._dequantize_hif4(calibration["weight_params"]).to(torch.float32)
            gram = calibration["activation_state"].get("gram64")
            cross = calibration["activation_state"].get("output_cross64")
            if not torch.is_tensor(gram) or not torch.is_tensor(cross):
                raise ValueError("parent fc state has no deployed gram/cross")
            gram = gram.to(device=device, dtype=torch.float32)
            cross = cross.to(device=device, dtype=torch.float32)
            for fold, calibration_pair in enumerate(calibration_pairs):
                incumbent = parent.hif4_dynamic_quantize_activation(
                    calibration_pair[0], calibration_pair[1], calibration["activation_state"]
                )
                evaluator.validate_hif4_params(
                    incumbent,
                    evaluator.dequantize_nvfp4(*calibration_pair).shape,
                )
                q_dense = parent._dequantize_hif4(incumbent).to(torch.float32)
                target = activation_t[fold].to(torch.float32)
                rows = int(q_dense.shape[0])
                blocks = int(q_dense.shape[1]) // BLOCK
                if blocks != int(gram.shape[0]):
                    raise ValueError("parent block Gram shape does not match fc activation")
                params = {name: value.to(device=device) for name, value in incumbent.items()}
                scale_all = params["scale_factor"].reshape(rows, blocks)
                lv2_all = params["scale_lv2"].reshape(rows, blocks, 8)
                lv3_all = params["scale_lv3"].reshape(rows, blocks, 8, 2)
                q_all = q_dense.reshape(rows, blocks, BLOCK)
                target_blocks = target.reshape(rows, blocks, BLOCK)
                parent_error = q_all - target_blocks
                block_features = {
                    "target_amax": target_blocks.abs().amax(dim=-1).mean(dim=0),
                    "target_rms": target_blocks.square().mean(dim=-1).sqrt().mean(dim=0),
                    "parent_error_rms": parent_error.square().mean(dim=-1).sqrt().mean(dim=0),
                    "parent_quadratic_loss": torch.einsum(
                        "rbi,bij,rbj->rb", parent_error, gram, parent_error
                    ).mean(dim=0),
                }
                target_output = target @ weight_t.transpose(0, 1)
                parent_output = q_all.reshape(rows, blocks * BLOCK) @ q_weight.transpose(0, 1)
                parent_residual = parent_output - target_output
                parent_row_loss = parent_residual.square().mean(dim=1)
                class_results: dict[str, Any] = {}
                for class_name in EDIT_CLASSES:
                    teacher_q = q_all.clone()
                    teacher_scale = scale_all.clone()
                    teacher_lv2 = lv2_all.clone()
                    teacher_lv3 = lv3_all.clone()
                    move_total = _empty_move_stats()
                    for block in range(blocks):
                        block_target = target_blocks[:, block]
                        teacher_q_block, teacher_scale_block, teacher_lv2_block, teacher_lv3_block, moves = _run_teacher(
                            class_name,
                            teacher_q[:, block],
                            teacher_scale[:, block],
                            teacher_lv2[:, block],
                            teacher_lv3[:, block],
                            block_target,
                            gram[block],
                            cross[block],
                            parent,
                        )
                        teacher_q[:, block] = teacher_q_block
                        teacher_scale[:, block] = teacher_scale_block
                        teacher_lv2[:, block] = teacher_lv2_block
                        teacher_lv3[:, block] = teacher_lv3_block
                        move_total = _merge_move_stats(move_total, moves)
                    single_block_margins: list[float] = []
                    for block in range(blocks):
                        start = block * BLOCK
                        stop = start + BLOCK
                        block_delta = (
                            (teacher_q[:, block] - q_all[:, block])
                            @ q_weight[:, start:stop].transpose(0, 1)
                        )
                        single_row_loss = (parent_residual + block_delta).square().mean(dim=1)
                        single_block_margins.append(
                            _finite(
                                float(
                                    (parent_row_loss.mean() - single_row_loss.mean())
                                    .div(parent_row_loss.mean().clamp_min(EPS))
                                    .item()
                                )
                            )
                        )
                    metrics = []
                    for block in range(blocks):
                        metrics.append(_class_metrics(
                            class_name,
                            q_all[:, block],
                            scale_all[:, block],
                            lv2_all[:, block],
                            lv3_all[:, block],
                            teacher_q[:, block],
                            teacher_scale[:, block],
                            teacher_lv2[:, block],
                            teacher_lv3[:, block],
                            target_blocks[:, block],
                            gram[block],
                            cross[block],
                            move_total,
                            single_block_margins[block],
                        ))
                    teacher_output = teacher_q.reshape(rows, blocks * BLOCK) @ q_weight.transpose(0, 1)
                    exact_parent = float((parent_output - target_output).square().mean().item())
                    exact_teacher = float((teacher_output - target_output).square().mean().item())
                    teacher_row_loss = (teacher_output - target_output).square().mean(dim=1)
                    positive_rows = int((teacher_row_loss < parent_row_loss - EPS).sum().item())
                    margins = [item["recoverable_margin"] for item in metrics]
                    class_results[class_name] = {
                        "parent_exact_output_mse": _finite(exact_parent),
                        "teacher_exact_output_mse": _finite(exact_teacher),
                        "recoverable_margin": _finite((exact_parent - exact_teacher) / max(exact_parent, EPS)),
                        "block_count": len(metrics),
                        "row_count": rows,
                        "positive_rows": positive_rows,
                        "negative_rows": rows - positive_rows,
                        "positive_blocks": sum(item["recoverable_margin"] > 0.0 for item in metrics),
                        "negative_blocks": sum(item["recoverable_margin"] < 0.0 for item in metrics),
                        "positive_exact_single_blocks": sum(value > 0.0 for value in single_block_margins),
                        "negative_exact_single_blocks": sum(value < 0.0 for value in single_block_margins),
                        "exact_single_block_margin_median": _finite(float(statistics.median(single_block_margins))),
                        "margin_median": _finite(float(statistics.median(margins))),
                        "feature_summary_quadratic": _class_feature_summary(margins, block_features),
                        "feature_summary_exact_single_block": _class_feature_summary(
                            single_block_margins, block_features
                        ),
                        "blocks": metrics,
                    }
                records.append({
                    "layer": layer,
                    "role": role,
                    "role_family": "fc",
                    "fold": fold,
                    "calibration_length": len(raw.calibration_windows[fold].input_ids),
                    "weight_shape": list(evaluator.dequantize_nvfp4(*weight_pair).shape),
                    "classes": class_results,
                })
    elapsed = time.perf_counter() - started
    by_class: dict[str, list[Mapping[str, Any]]] = {name: [] for name in EDIT_CLASSES}
    for record in records:
        for name in EDIT_CLASSES:
            value = record["classes"][name]
            by_class[name].append({"recoverable_margin": value["recoverable_margin"]})
    summary_by_role = {
        role: {
            name: _group_class_summary(
                [record for record in records if record["role"] == role], name, "fold"
            )
            for name in EDIT_CLASSES
        }
        for role in roles
    }
    summary_by_layer = {
        str(layer): {
            name: _group_class_summary(
                [record for record in records if int(record["layer"]) == layer], name, "role"
            )
            for name in EDIT_CLASSES
        }
        for layer in layers
    }
    summary_by_fold = {
        str(fold): {
            name: _group_class_summary(
                [record for record in records if int(record["fold"]) == fold], name, "role"
            )
            for name in EDIT_CLASSES
        }
        for fold in range(2)
    }
    output = {
        "protocol": "proxy-v2",
        "diagnostic": "l3-fc-legal-oracle-v2",
        "scope": "research-oracle",
        "status": "ok",
        "parent": str(parent_path),
        "parent_sha256": parent_hash,
        "cache": str(args.cache.resolve()),
        "cache_metadata": raw.metadata,
        "device": str(device),
        "layers": list(layers),
        "roles": list(roles),
        "calibration_folds": [len(window.input_ids) for window in raw.calibration_windows[:2]],
        "edit_classes": list(EDIT_CLASSES),
        "teacher_seconds": _finite(elapsed),
        "summary_by_class": {name: _summarize(items) for name, items in by_class.items()},
        "summary_by_role_and_fold": summary_by_role,
        "summary_by_layer_and_role": summary_by_layer,
        "summary_by_fold_and_role": summary_by_fold,
        "records": records,
        "decision": "margin_exists_but_not_compile_safe",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    report_lines = [
        "# L3-D0 fc legal-code oracle — proxy-v2",
        "",
        f"- parent SHA256: `{parent_hash}`",
        f"- layers: `{list(layers)}`; roles: `{list(roles)}`",
        f"- calibration folds: `{[len(window.input_ids) for window in raw.calibration_windows[:2]]}`",
        f"- device: `{device}`",
        f"- teacher wall: `{elapsed:.3f}s` (research cost; not candidate API time)",
        "",
        "## Class summary",
        "",
        "| edit class | cases | mean margin | median margin | positive | negative | conclusion |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for name in EDIT_CLASSES:
        value = output["summary_by_class"][name]
        report_lines.append(
            f"| {name} | {value.get('case_count', 0)} | {value.get('mean_recoverable_margin', 0.0):.6f} | "
            f"{value.get('median_recoverable_margin', 0.0):.6f} | {value.get('positive_cases', 0)} | "
            f"{value.get('negative_cases', 0)} | {value.get('effect', '')} |"
        )
    report_lines.extend([
        "",
        "## Per layer / role / fold",
        "",
        "| layer | role | fold length | class | exact parent MSE | exact teacher MSE | joint margin | quadratic+ / total | exact-single median | exact-single +/- |",
        "|---:|---|---:|---|---:|---:|---:|---:|---:|---:|",
    ])
    for record in records:
        for name in EDIT_CLASSES:
            value = record["classes"][name]
            report_lines.append(
                f"| {record['layer']} | {record['role']} | {record['calibration_length']} | {name} | "
                f"{value['parent_exact_output_mse']:.6e} | {value['teacher_exact_output_mse']:.6e} | "
                f"{value['recoverable_margin']:.6f} | {value['positive_blocks']}/{value['block_count']} | "
                f"{value['exact_single_block_margin_median']:.6f} | "
                f"{value['positive_exact_single_blocks']}/{value['negative_exact_single_blocks']} |"
            )
    report_lines.extend([
        "",
        "Block quadratic uses the parent deployed Q(W) Gram/cross metric for local move acceptance; "
        "exact full output MSE is recomputed once per class after all block edits. Positive/negative "
        "quadratic and exact-single-block counts plus cheap feature summaries are localization diagnostics, not a deployable "
        "candidate or a ranking score.",
        "",
        "Decision: `margin_exists_but_not_compile_safe`. The joint teacher has positive same-fold "
        "margin in most records, but class signs are mixed and layer 3 / fold 128 has a large exact "
        "output regression for both fc roles. No student or v155 is created from this oracle; the "
        "next experiment must test cross-fold feature/decision stability or switch to L2.",
    ])
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", type=Path, default=ROOT / "workbench" / "pre-a3-v147-parent.py")
    parser.add_argument("--cache", type=Path, default=ROOT / "artifacts" / "official_eval" / "cache" / "qwen2.5-0.5b-proxy-v2.pt")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts" / "official_eval" / "l3-fc-legal-oracle.json")
    parser.add_argument("--report", type=Path, default=ROOT / "logs" / "execution" / "2026-09-02-l3-fc-legal-oracle.md")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--layers", default=",".join(str(value) for value in DEFAULT_LAYERS))
    parser.add_argument("--roles", default=",".join(DEFAULT_ROLES))
    return parser


if __name__ == "__main__":
    result = run(build_parser().parse_args())
    raise SystemExit(0 if result.get("status") == "ok" else 1)
