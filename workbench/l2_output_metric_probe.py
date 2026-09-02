"""Local-only L2 output-metric-gated analytic pair transform.

The first L2 probe balanced continuous activation/weight moments and then let
the parent quantizer decide.  That changed the static Weight code too
aggressively and regressed every focused fc case.  This probe keeps the same
single analytic 2x2 construction, but damps it deterministically and admits it
only when the *quantized product* on both calibration folds improves:

    ||X W.T - Q(X M) Q(W M^{-T}).T||^2

The transform is evaluated once per expansive matrix.  There is no angle,
alpha, seed, or block-size sweep.  This module imports an immutable parent and
is therefore a research control, not a submission candidate.
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
spec = importlib.util.spec_from_file_location("l2_output_metric_parent", PARENT_PATH)
if spec is None or spec.loader is None:
    raise ImportError(f"cannot load parent source: {PARENT_PATH}")
parent = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = parent
spec.loader.exec_module(parent)
if hashlib.sha256(PARENT_PATH.read_bytes()).hexdigest().lower() != PARENT_SHA256:
    raise RuntimeError("L2 output-metric probe parent SHA mismatch")

BLOCK = 64
PAIR = 2
SAMPLE_ROWS = 128
EPS = 1.0e-12

_ORIGINAL_LEARN_ROAB = parent._learn_roab_pairs
_LAST_DIAGNOSTIC: dict[str, Any] = {}


def _sym_root(matrices: torch.Tensor, *, inverse: bool = False) -> torch.Tensor:
    sym = 0.5 * (matrices + matrices.transpose(-1, -2))
    eye = torch.eye(PAIR, device=sym.device, dtype=sym.dtype).expand_as(sym)
    values, vectors = torch.linalg.eigh(sym + float(parent._ROAB_PAIR_RIDGE) * eye)
    values = values.clamp_min(float(parent._ROAB_PAIR_RIDGE))
    values = values.rsqrt() if inverse else values.sqrt()
    return (vectors * values.unsqueeze(-2)).matmul(vectors.transpose(-1, -2))


def _analytic_damped_pair(
    weight: torch.Tensor,
    calibration: Sequence[torch.Tensor],
) -> torch.Tensor | None:
    """Construct one deterministic, close-to-identity pair transform."""
    rows, channels = map(int, weight.shape)
    if (
        weight.ndim != 2
        or rows <= channels
        or channels % BLOCK != 0
        or channels % PAIR != 0
        or not calibration
    ):
        return None
    try:
        activation = torch.cat(
            [parent._sample_rows(item, SAMPLE_ROWS).to(torch.float32) for item in calibration[:2]],
            dim=0,
        )
        weight_sample = parent._sample_rows(weight, SAMPLE_ROWS * 2).to(torch.float32)
        pairs = channels // PAIR
        a_blocks = activation.reshape(-1, pairs, PAIR)
        w_blocks = weight_sample.reshape(-1, pairs, PAIR)
        ga = torch.einsum("npi,npj->pij", a_blocks, a_blocks)
        gw = torch.einsum("npi,npj->pij", w_blocks, w_blocks)
        ga = ga / max(1, int(activation.shape[0]))
        gw = gw / max(1, int(weight_sample.shape[0]))

        # The same SPD moment balance as the rejected probe is only a direction
        # proposal.  The output-metric score below is the actual admission rule.
        a_root = _sym_root(ga)
        a_inv_root = _sym_root(ga, inverse=True)
        middle = a_root.matmul(gw).matmul(a_root)
        balance = a_inv_root.matmul(_sym_root(middle)).matmul(a_inv_root)
        full = _sym_root(balance)

        eye = torch.eye(PAIR, device=full.device, dtype=full.dtype).expand_as(full)
        delta = full - eye
        singular_delta = torch.linalg.svdvals(delta)[..., 0]
        # No tuned damping grid: normalize the proposed displacement by its
        # own spectral norm so the deployed code sees a bounded transform.
        eta = (1.0 / (1.0 + singular_delta)).reshape(-1, 1, 1)
        matrix = eye + eta * delta
        if not bool(torch.isfinite(matrix).all()):
            return None
        # A second, fixed condition-number clamp keeps the reciprocal transform
        # close enough to the parent code frame that a calibration-window
        # improvement cannot be purchased by a large static-code relocation.
        left, singular, right_t = torch.linalg.svd(matrix)
        if not bool(torch.isfinite(singular).all()):
            return None
        singular = singular.clamp(0.75, 4.0 / 3.0)
        matrix = left.matmul(torch.diag_embed(singular)).matmul(right_t)
        return matrix.contiguous()
    except (RuntimeError, ValueError, FloatingPointError):
        return None


def _quantized_output_score(
    weight: torch.Tensor,
    calibration: Sequence[torch.Tensor],
    matrix: torch.Tensor | None,
) -> tuple[float, list[float]]:
    """Score the legal quantized product on both calibration folds."""
    if matrix is None:
        transform = None
    else:
        transform = matrix.to(device=weight.device, dtype=torch.float32)
    try:
        if transform is None:
            transformed_weight = weight.to(torch.float32)
            transformed_acts = [item.to(torch.float32) for item in calibration[:2]]
        else:
            inverse_t = torch.linalg.inv(transform).transpose(-1, -2)
            transformed_weight = parent._pair_transform(weight, inverse_t)
            transformed_acts = [parent._pair_transform(item, transform) for item in calibration[:2]]
        weight_sample = parent._sample_rows(transformed_weight, SAMPLE_ROWS)
        weight_params = parent._dense_to_hif4(weight_sample, offsets=parent._BASE_OFFSETS)
        quantized_weight = parent._dequantize_hif4(weight_params).to(torch.float32)
        # The sampled weight rows above are only valid if the original tensor
        # has the same row count.  Re-encode all rows for the actual product.
        weight_params = parent._dense_to_hif4(transformed_weight, offsets=parent._BASE_OFFSETS)
        quantized_weight = parent._dequantize_hif4(weight_params).to(torch.float32)
        losses: list[float] = []
        for original, transformed in zip(calibration[:2], transformed_acts):
            original = parent._sample_rows(original, SAMPLE_ROWS).to(torch.float32)
            transformed = parent._sample_rows(transformed, SAMPLE_ROWS).to(torch.float32)
            activation_params = parent._dense_to_hif4(transformed, offsets=parent._BASE_OFFSETS)
            quantized_activation = parent._dequantize_hif4(activation_params).to(torch.float32)
            raw_weight = parent._sample_rows(weight, int(quantized_weight.shape[0])).to(torch.float32)
            q_weight = parent._sample_rows(quantized_weight, int(raw_weight.shape[0])).to(torch.float32)
            target = original.mm(raw_weight.t())
            predicted = quantized_activation.mm(q_weight.t())
            losses.append(float((predicted - target).square().mean().div(target.square().mean().clamp_min(EPS)).item()))
        if not losses:
            return math.inf, []
        robust = sum(losses) / len(losses) + 0.25 * max(losses)
        return float(robust), losses
    except (RuntimeError, ValueError, FloatingPointError):
        return math.inf, []


def _learn_roab_output_metric(
    weight: torch.Tensor,
    calibration: Sequence[torch.Tensor],
) -> torch.Tensor | None:
    global _LAST_DIAGNOSTIC
    # Build each proposal from one fold and admit it only on the other fold.
    # This is the same cross-fit principle used by the parent Weight HSDQ pass
    # and prevents a small same-fold output gain from selecting a transform that
    # fails on the held-out calibration window.
    folds = list(calibration[:2])
    base_score, base_losses = _quantized_output_score(weight, folds, None)
    candidates: list[tuple[str, torch.Tensor | None]] = []
    for index, fold in enumerate(folds):
        candidates.append((f"fold{index}", _analytic_damped_pair(weight, [fold])))
    heldout_losses: list[float | None] = []
    for index, (_, candidate) in enumerate(candidates):
        if candidate is None:
            heldout_losses.append(None)
            continue
        heldout_fold = [folds[1 - index]]
        _, loss = _quantized_output_score(weight, heldout_fold, candidate)
        heldout_losses.append(None if not loss else float(loss[0]))
    # Require every fold-specific proposal to improve the opposite fold.  If
    # either direction is not supported, return identity and leave the parent
    # BOAT/CAT path untouched.
    crossfit_ok = (
        len(candidates) == 2
        and all(candidate is not None for _, candidate in candidates)
        and all(
            heldout_losses[index] is not None
            and float(heldout_losses[index]) < float(base_losses[1 - index])
            for index in range(2)
        )
    )
    if crossfit_ok:
        scored: list[tuple[float, str, torch.Tensor]] = []
        for name, candidate in candidates:
            assert candidate is not None
            score, _ = _quantized_output_score(weight, folds, candidate)
            if math.isfinite(score):
                scored.append((float(score), name, candidate))
        scored.sort(key=lambda item: (item[0], item[1]))
        selected = scored[0] if scored else None
    else:
        selected = None
    candidate = None if selected is None else selected[2]
    candidate_score, candidate_losses = _quantized_output_score(weight, folds, candidate)
    accepted = bool(crossfit_ok and candidate is not None and math.isfinite(candidate_score) and candidate_score < base_score)
    _LAST_DIAGNOSTIC = {
        "expansive": bool(int(weight.shape[0]) > int(weight.shape[1])),
        "base_score": float(base_score),
        "candidate_score": float(candidate_score),
        "base_fold_losses": [float(value) for value in base_losses],
        "candidate_fold_losses": [float(value) for value in candidate_losses],
        "crossfit_heldout_losses": [None if value is None else float(value) for value in heldout_losses],
        "crossfit_ok": bool(crossfit_ok),
        "selected_candidate": None if selected is None else str(selected[1]),
        "accepted": bool(accepted),
        "candidate": "crossfit_damped_analytic_pair" if candidate is not None else "none",
    }
    return candidate if accepted else None


parent._learn_roab_pairs = _learn_roab_output_metric


def hif4_calibration_and_quantize_weight(
    weight_quant: torch.Tensor,
    weight_scale: torch.Tensor,
    calib_activation_list: list,
) -> dict[str, Any]:
    result = parent.hif4_calibration_and_quantize_weight(
        weight_quant, weight_scale, calib_activation_list
    )
    if _LAST_DIAGNOSTIC:
        state = dict(result["activation_state"])
        state["l2_output_metric_pair"] = bool(_LAST_DIAGNOSTIC.get("accepted", False))
        state["l2_output_metric_diagnostic"] = dict(_LAST_DIAGNOSTIC)
        state["version"] = 5
        result["activation_state"] = state
    return result


def hif4_dynamic_quantize_activation(
    activation_quant: torch.Tensor,
    activation_scale: torch.Tensor,
    activation_state: Any,
) -> dict[str, torch.Tensor]:
    return parent.hif4_dynamic_quantize_activation(
        activation_quant, activation_scale, activation_state
    )


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
