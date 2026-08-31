"""Offline JDRQ ceiling/quantization-gap diagnostic.

This tool is deliberately outside the six submission APIs.  It loads one
cached model layer, freezes the normal activation state, and reports:

* the parent static-Q(W) product loss;
* a continuous dual-ridge target loss (an upper-bound direction);
* the legal HiF4 projection loss;
* the fixed-Q(A) hierarchy residual loss.

It never writes a product or residual into an activation state and never
reads evaluator test windows.  The output is a development diagnostic, not an
official-score substitute.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
EVALUATOR = Path(__file__).resolve().parent
if str(EVALUATOR) not in sys.path:
    sys.path.insert(0, str(EVALUATOR))

from nvfp4_sim import nvfp4_encode  # noqa: E402
import solution  # noqa: E402


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--role", required=True)
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--mode", choices=("amax6", "amax4", "pow2"), default="amax6")
    parser.add_argument("--rows", type=int, default=256)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def _state_transform(state: dict, channels: int, device: torch.device):
    smooth_inv = state.get("smooth_inv")
    d = (
        torch.ones(channels, dtype=torch.float32, device=device)
        if smooth_inv is None
        else smooth_inv.to(device=device, dtype=torch.float32).reciprocal()
    )
    permutation = state.get("permutation")
    if permutation is None:
        permutation = solution._identity_permutation(channels, device)
    else:
        permutation = permutation.to(device=device, dtype=torch.int64)
    return (
        d,
        permutation,
        int(state.get("block_smooth_size", 0)),
        int(state.get("block_smooth_seed", 0)),
        state.get("cat_transform"),
    )


def run(args: argparse.Namespace) -> dict:
    payload = torch.load(args.cache, map_location="cpu", weights_only=False)
    layer = int(args.layer)
    role = str(args.role)
    if layer < 0 or layer >= len(payload["weights"]):
        raise ValueError(f"layer {layer} is outside cached layer range")
    if role not in payload["weights"][layer]:
        raise ValueError(f"role {role!r} is not present in cache")

    # The diagnostic starts from the ordinary parent, not from a previously
    # selected JDRQ candidate.  This isolates the measured mechanisms.
    solution._JDRQ_ENABLED = False
    solution._JDRQ_DUAL_ENABLED = False
    weight = payload["weights"][layer][role].to(torch.float32)
    weight_pair = nvfp4_encode(weight, args.mode)
    calibration_pairs = [
        nvfp4_encode(
            payload["calibration_activations"][role][batch][layer], args.mode
        )
        for batch in range(len(payload["calibration_windows"]))
    ]
    calibrated = solution.hif4_calibration_and_quantize_weight(
        *weight_pair, calibration_pairs
    )
    state = calibrated["activation_state"]
    parent = calibrated["weight_params"]
    device = weight.device
    d, permutation, block_size, block_seed, cat_transform = _state_transform(
        state, int(weight.shape[1]), device
    )
    # Match the evaluator's reference exactly: calibration receives NVFP4
    # weight inputs, so the teacher must use the supplied NVFP4 dequantization,
    # not the uncoded cache tensor.
    weight_reference = solution._dequantize_nvfp4_float32(*weight_pair)
    transformed_weight = solution._linear_pair_transform(
        weight_reference,
        d,
        permutation,
        block_size,
        block_seed,
        weight_side=True,
        cat_transform=cat_transform,
    )
    exact_activation, frozen_activation, boundaries = (
        solution._jdrq_calibration_products(
            calibration_pairs,
            state,
            d,
            permutation,
            block_size,
            block_seed,
            cat_transform,
            max_rows=max(1, int(args.rows)),
        )
    )
    teacher = exact_activation.mm(transformed_weight.t())
    parent_loss = solution._jdrq_robust_product_loss(
        frozen_activation, teacher, parent, boundaries
    )
    parent_dense = solution._dequantize_hif4(parent)
    continuous_target = solution._jdrq_make_target(
        frozen_activation,
        teacher,
        parent_dense,
        0.10,
        1.0,
    )
    continuous_loss = float(
        (teacher - frozen_activation.mm(continuous_target.t())).square().mean()
        / teacher.square().mean().clamp_min(solution._EPS)
    )
    legal_params = solution._dense_to_hif4(
        continuous_target,
        search_offsets=solution._WEIGHT_OFFSETS,
        error_threshold=solution._WEIGHT_REFINE_ERROR_THRESHOLD,
        accept_margin=solution._WEIGHT_REFINE_ACCEPT_MARGIN,
        max_refine_ratio=solution._WEIGHT_REFINE_MAX_RATIO_SMALL,
        max_refine_blocks=solution._WEIGHT_REFINE_MAX_BLOCKS,
    )
    legal_loss = solution._jdrq_robust_product_loss(
        frozen_activation, teacher, legal_params, boundaries
    )
    hierarchy_params = solution._jdrq_refine_hierarchy_offsets(
        frozen_activation,
        teacher,
        transformed_weight,
        parent,
        max_ratio=solution._JDRQ_HIERARCHY_RATIO,
        max_blocks=solution._JDRQ_HIERARCHY_MAX_BLOCKS,
        offsets=solution._JDRQ_HIERARCHY_OFFSETS,
    )
    hierarchy_loss = solution._jdrq_robust_product_loss(
        frozen_activation, teacher, hierarchy_params, boundaries
    )
    result = {
        "cache": str(args.cache),
        "layer": layer,
        "role": role,
        "mode": args.mode,
        "weight_shape": list(weight.shape),
        "calibration_rows": int(frozen_activation.shape[0]),
        "calibration_windows": [list(item) for item in boundaries],
        "parent_product_loss": parent_loss,
        "continuous_dual_loss": continuous_loss,
        "legal_projection_loss": legal_loss,
        "hierarchy_residual_loss": hierarchy_loss,
        "continuous_gap_ratio": continuous_loss / max(parent_loss, solution._EPS),
        "legal_gap_ratio": legal_loss / max(parent_loss, solution._EPS),
        "hierarchy_ratio": hierarchy_loss / max(parent_loss, solution._EPS),
        "jdrq_default": bool(solution._JDRQ_ENABLED),
        "state_keys": sorted(state.keys()),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    return result


if __name__ == "__main__":
    arguments = _parser().parse_args()
    print(json.dumps(run(arguments), ensure_ascii=False, indent=2))
