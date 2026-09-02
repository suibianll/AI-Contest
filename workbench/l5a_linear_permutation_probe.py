"""Local-only Linear hierarchy-aware permutation probe.

The current pre-A3 parent has Attention permutations but no Linear channel
permutation.  Historical L5a evidence suggests that rearranging pressure
within each 64-channel HiF4 block can reduce shared hierarchy-scale conflict.
This control tests one fixed low/high interleave for expansive fc shapes only.
The permutation is admitted by a two-fold real quantized-product score and is
stored only in the local state; no candidate list or dynamic search is added.
"""

from __future__ import annotations

import hashlib
import importlib.util
import math
import sys
from pathlib import Path
from typing import Any, Sequence

import torch

ROOT = Path(__file__).resolve().parents[1]
PARENT_PATH = ROOT / "workbench" / "pre-a3-v147-parent.py"
PARENT_SHA256 = "800ca10ec3414e4fe886b93ca62bd4a350d26bba015287df7e8df2dd871ac23d"
spec = importlib.util.spec_from_file_location("l5a_linear_permutation_parent", PARENT_PATH)
if spec is None or spec.loader is None:
    raise ImportError(f"cannot load source: {PARENT_PATH}")
parent = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = parent
spec.loader.exec_module(parent)
if hashlib.sha256(PARENT_PATH.read_bytes()).hexdigest().lower() != PARENT_SHA256:
    raise RuntimeError("L5a probe parent SHA mismatch")

BLOCK = 64
SAMPLE_ROWS = 128
EPS = 1.0e-12
_ORIGINAL_APPLY = parent._apply_boat_rotation
_ORIGINAL_CHOOSE = parent._choose_boat
_ACTIVE_PERM: torch.Tensor | None = None
_LAST_DIAGNOSTIC: dict[str, Any] = {}


def _identity(channels: int, device: torch.device) -> torch.Tensor:
    return torch.arange(channels, device=device, dtype=torch.int64)


def _pressure(
    weight: torch.Tensor,
    calibration: Sequence[torch.Tensor],
    balance: torch.Tensor,
) -> torch.Tensor:
    joined = torch.cat([parent._sample_rows(item, SAMPLE_ROWS) for item in calibration[:2]], dim=0).to(torch.float32)
    d = balance.to(device=weight.device, dtype=torch.float32).reshape(1, -1)
    w = parent._sample_rows(weight, SAMPLE_ROWS * 2).to(torch.float32) * d
    a = joined / d
    w_rms = w.square().mean(dim=0).add(EPS).sqrt()
    a_rms = a.square().mean(dim=0).add(EPS).sqrt()
    pressure = 0.5 * (
        torch.log1p((w.abs().amax(dim=0) / w_rms).clamp_min(0.0))
        + torch.log1p((a.abs().amax(dim=0) / a_rms).clamp_min(0.0))
    )
    return torch.nan_to_num(pressure, nan=0.0, posinf=32.0, neginf=0.0)


def _low_high_interleave(pressure: torch.Tensor) -> torch.Tensor:
    channels = int(pressure.numel())
    order = _identity(channels, pressure.device)
    for start in range(0, channels, BLOCK):
        local = torch.argsort(pressure[start : start + BLOCK], stable=True)
        low = local[: BLOCK // 2]
        high = local[BLOCK // 2 :].flip(0)
        chosen = torch.empty_like(local)
        chosen[0::2] = low
        chosen[1::2] = high
        order[start : start + BLOCK] = chosen + start
    return order


def _apply_with_order(
    value: torch.Tensor,
    order: torch.Tensor | None,
    seed: int,
    block_size: int,
) -> torch.Tensor:
    if order is not None:
        value = value.index_select(-1, order.to(device=value.device))
    return _ORIGINAL_APPLY(value, seed, block_size)


def _product_score(
    weight: torch.Tensor,
    calibration: Sequence[torch.Tensor],
    balance: torch.Tensor,
    seed: int,
    block_size: int,
    order: torch.Tensor | None,
) -> tuple[float, list[float]]:
    try:
        weight_t = _apply_with_order(weight * balance.reshape(1, -1), order, seed, block_size)
        q_weight = parent._dequantize_hif4(
            parent._dense_to_hif4(weight_t, offsets=parent._BASE_OFFSETS)
        ).to(torch.float32)
        losses: list[float] = []
        for sample in calibration[:2]:
            sample = parent._sample_rows(sample, SAMPLE_ROWS).to(torch.float32)
            activation_t = _apply_with_order(sample / balance.reshape(1, -1), order, seed, block_size)
            q_activation = parent._dequantize_hif4(
                parent._dense_to_hif4(activation_t, offsets=parent._BASE_OFFSETS)
            ).to(torch.float32)
            target = activation_t.mm(weight_t.t())
            predicted = q_activation.mm(q_weight.t())
            losses.append(float((predicted - target).square().mean().div(target.square().mean().clamp_min(EPS)).item()))
        if not losses:
            return math.inf, []
        return sum(losses) / len(losses) + 0.25 * max(losses), losses
    except (RuntimeError, ValueError, FloatingPointError):
        return math.inf, []


def _choose_boat_with_perm(
    weight: torch.Tensor,
    calibration: Sequence[torch.Tensor],
):
    global _ACTIVE_PERM, _LAST_DIAGNOSTIC
    _ACTIVE_PERM = None
    balance, seed, block_size = _ORIGINAL_CHOOSE(weight, calibration)
    rows, channels = map(int, weight.shape)
    diagnostic: dict[str, Any] = {
        "shape": [rows, channels],
        "expansive": bool(rows > channels),
        "accepted": False,
    }
    if rows <= channels or channels % BLOCK != 0 or not calibration:
        _LAST_DIAGNOSTIC = diagnostic
        return balance, seed, block_size
    try:
        pressure = _pressure(weight, calibration, balance)
        candidate = _low_high_interleave(pressure)
        base_score, base_losses = _product_score(
            weight, calibration, balance, seed, block_size, None
        )
        candidate_score, candidate_losses = _product_score(
            weight, calibration, balance, seed, block_size, candidate
        )
        accepted = (
            math.isfinite(candidate_score)
            and math.isfinite(base_score)
            and candidate_score < base_score
            and len(base_losses) == len(candidate_losses) == 2
            and all(c <= b for c, b in zip(candidate_losses, base_losses))
        )
        if accepted:
            _ACTIVE_PERM = candidate
        diagnostic.update({
            "base_score": float(base_score),
            "candidate_score": float(candidate_score),
            "base_fold_losses": [float(value) for value in base_losses],
            "candidate_fold_losses": [float(value) for value in candidate_losses],
            "accepted": bool(accepted),
            "changed_channels": int((candidate != _identity(channels, weight.device)).sum().item()),
        })
    except (RuntimeError, ValueError, FloatingPointError) as exc:
        diagnostic["error"] = f"{type(exc).__name__}: {exc}"
    _LAST_DIAGNOSTIC = diagnostic
    return balance, seed, block_size


def _apply_with_active(
    value: torch.Tensor,
    seed: int,
    block_size: int = parent._BLOCK,
) -> torch.Tensor:
    return _apply_with_order(value, _ACTIVE_PERM, seed, block_size)


parent._choose_boat = _choose_boat_with_perm
parent._apply_boat_rotation = _apply_with_active


def hif4_calibration_and_quantize_weight(
    weight_quant: torch.Tensor,
    weight_scale: torch.Tensor,
    calib_activation_list: list,
) -> dict[str, Any]:
    global _ACTIVE_PERM
    _ACTIVE_PERM = None
    result = parent.hif4_calibration_and_quantize_weight(
        weight_quant, weight_scale, calib_activation_list
    )
    state = dict(result["activation_state"])
    if _ACTIVE_PERM is not None:
        state["linear_permutation"] = _ACTIVE_PERM.detach().to(device="cpu", dtype=torch.int32)
        state["l5a_linear_permutation"] = True
        state["version"] = 7
    result["activation_state"] = state
    return result


def hif4_dynamic_quantize_activation(
    activation_quant: torch.Tensor,
    activation_scale: torch.Tensor,
    activation_state: Any,
) -> dict[str, torch.Tensor]:
    global _ACTIVE_PERM
    previous = _ACTIVE_PERM
    value = activation_state.get("linear_permutation") if isinstance(activation_state, dict) else None
    _ACTIVE_PERM = value.to(device=activation_quant.device, dtype=torch.int64) if torch.is_tensor(value) else None
    try:
        return parent.hif4_dynamic_quantize_activation(
            activation_quant, activation_scale, activation_state
        )
    finally:
        _ACTIVE_PERM = previous


hif4_calibration_attention = parent.hif4_calibration_attention
hif4_dynamic_quantize_q = parent.hif4_dynamic_quantize_q
hif4_dynamic_quantize_k = parent.hif4_dynamic_quantize_k
hif4_dynamic_quantize_v = parent.hif4_dynamic_quantize_v

__all__ = [
    "hif4_calibration_and_quantize_weight",
    "hif4_dynamic_quantize_activation",
    "hif4_calibration_attention",
    "hif4_dynamic_quantize_q",
    "hif4_dynamic_quantize_k",
    "hif4_dynamic_quantize_v",
]
