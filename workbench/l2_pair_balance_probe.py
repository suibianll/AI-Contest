"""Local-only L2 analytic pair-balance probe.

This wrapper replaces the parent's diagonal BOAT/CAT/ROAB selection only for
expansive Linear shapes (rows > channels).  It solves the calibration moment
equation ``S A S = B`` independently for each two-channel pair, uses the
symmetric square root ``M`` with ``M M.T = S``, and lets the unchanged parent
quantizers/refiners handle legal HiF4 coding.  Attention and non-expansive
roles delegate to the immutable parent.  The module is a research control,
not a formal candidate, because it imports the parent source.
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
spec = importlib.util.spec_from_file_location("l2_pair_parent", PARENT_PATH)
if spec is None or spec.loader is None:
    raise ImportError(f"cannot load parent source: {PARENT_PATH}")
parent = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = parent
spec.loader.exec_module(parent)
if hashlib.sha256(PARENT_PATH.read_bytes()).hexdigest().lower() != PARENT_SHA256:
    raise RuntimeError("L2 probe parent SHA mismatch")

_ORIGINAL_CHOOSE_BOAT = parent._choose_boat
_ORIGINAL_CHOOSE_CAT = parent._choose_expansive_cat_balance
_ORIGINAL_LEARN_ROAB = parent._learn_roab_pairs
_ORIGINAL_APPLY_BOAT = parent._apply_boat_rotation
_ACTIVE_MATRIX: torch.Tensor | None = None
_ACTIVE_EXPANSIVE = False
_APPLY_INDEX = 0


def _sym_root(matrices: torch.Tensor, *, inverse: bool = False) -> torch.Tensor:
    sym = 0.5 * (matrices + matrices.transpose(-1, -2))
    eye = torch.eye(2, device=sym.device, dtype=sym.dtype).expand_as(sym)
    values, vectors = torch.linalg.eigh(sym + float(parent._ROAB_PAIR_RIDGE) * eye)
    values = values.clamp_min(float(parent._ROAB_PAIR_RIDGE))
    values = values.rsqrt() if inverse else values.sqrt()
    return (vectors * values.unsqueeze(-2)).matmul(vectors.transpose(-1, -2))


def _analytic_pair_matrix(weight: torch.Tensor, calibration: Sequence[torch.Tensor]) -> torch.Tensor | None:
    if weight.ndim != 2 or int(weight.shape[0]) <= int(weight.shape[1]) or int(weight.shape[1]) % 2:
        return None
    channels = int(weight.shape[1])
    pairs = channels // 2
    try:
        activation = torch.cat(
            [parent._sample_rows(item, 256).to(torch.float32) for item in calibration], dim=0
        )
        weight_sample = parent._sample_rows(weight, 512).to(torch.float32)
        a_blocks = activation.reshape(-1, pairs, 2)
        w_blocks = weight_sample.reshape(-1, pairs, 2)
        ga = torch.einsum("npi,npj->pij", a_blocks, a_blocks) / max(1, int(activation.shape[0]))
        gb = torch.einsum("npi,npj->pij", w_blocks, w_blocks) / max(1, int(weight_sample.shape[0]))
        a_root = _sym_root(ga)
        a_inv_root = _sym_root(ga, inverse=True)
        middle = a_root.matmul(gb).matmul(a_root)
        balance = a_inv_root.matmul(_sym_root(middle)).matmul(a_inv_root)
        matrices = _sym_root(balance)
        left, singular, right_t = torch.linalg.svd(matrices)
        singular = singular.clamp(float(parent._ROAB_MIN_SINGULAR), float(parent._ROAB_MAX_SINGULAR))
        matrices = left.matmul(torch.diag_embed(singular)).matmul(right_t)
        if not bool(torch.isfinite(matrices).all()):
            return None
        return matrices.contiguous()
    except (RuntimeError, ValueError, FloatingPointError):
        return None


def _choose_boat_l2(weight: torch.Tensor, calibration: Sequence[torch.Tensor]):
    global _ACTIVE_MATRIX, _ACTIVE_EXPANSIVE, _APPLY_INDEX
    _ACTIVE_MATRIX = _analytic_pair_matrix(weight, calibration)
    _ACTIVE_EXPANSIVE = _ACTIVE_MATRIX is not None
    _APPLY_INDEX = 0
    if _ACTIVE_EXPANSIVE:
        return torch.ones(int(weight.shape[1]), device=weight.device), -1, 0
    return _ORIGINAL_CHOOSE_BOAT(weight, calibration)


def _choose_cat_l2(
    weight: torch.Tensor,
    calibration: Sequence[torch.Tensor],
    base_balance: torch.Tensor,
    seed: int,
    block_size: int,
):
    if _ACTIVE_EXPANSIVE:
        return base_balance
    return _ORIGINAL_CHOOSE_CAT(weight, calibration, base_balance, seed, block_size)


def _learn_roab_l2(weight: torch.Tensor, calibration: Sequence[torch.Tensor]):
    if _ACTIVE_EXPANSIVE:
        return None
    return _ORIGINAL_LEARN_ROAB(weight, calibration)


def _apply_boat_l2(tensor: torch.Tensor, seed: int, block_size: int = parent._BLOCK):
    global _APPLY_INDEX
    if not _ACTIVE_EXPANSIVE or _ACTIVE_MATRIX is None:
        return _ORIGINAL_APPLY_BOAT(tensor, seed, block_size)
    # Parent calibration calls this once for W and then once per calibration A.
    # The first call is the static weight and needs M^{-T}; all later calls are
    # activations and need M.  The state stores M for dynamic calls, so this
    # ordering is local to the parent calibration transaction.
    _APPLY_INDEX += 1
    matrices = _ACTIVE_MATRIX.to(device=tensor.device, dtype=torch.float32)
    if _APPLY_INDEX == 1:
        matrices = torch.linalg.inv(matrices).transpose(-1, -2)
    return parent._pair_transform(tensor, matrices)


parent._choose_boat = _choose_boat_l2
parent._choose_expansive_cat_balance = _choose_cat_l2
parent._learn_roab_pairs = _learn_roab_l2
parent._apply_boat_rotation = _apply_boat_l2


def hif4_calibration_and_quantize_weight(
    weight_quant: torch.Tensor,
    weight_scale: torch.Tensor,
    calib_activation_list: list,
) -> dict[str, Any]:
    result = parent.hif4_calibration_and_quantize_weight(
        weight_quant, weight_scale, calib_activation_list
    )
    if _ACTIVE_EXPANSIVE and _ACTIVE_MATRIX is not None:
        state = dict(result["activation_state"])
        state["roab_pairs"] = _ACTIVE_MATRIX.detach().to(device="cpu", dtype=torch.float32).contiguous()
        state["l2_pair_balance"] = True
        state["version"] = 4
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
