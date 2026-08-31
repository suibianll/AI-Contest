"""Measure the legal E6M2 scale-lattice gap on cached Linear operands.

This is an evaluator-side research oracle, not a submission algorithm.  It
compares the current ``±3`` scale candidates with all 255 finite unsigned
E6M2 codes after the active BOAT transform.  Activation scoring uses only a
static weight Gram matrix; it never forms ``A @ W`` or routes an output into
``activation_state``.

Typical use::

    python evaluator/e6m2_scale_lattice_oracle.py \
        --cache artifacts/real_model_suite/cache/qwen2.5-0.5b__seq128__calib2__test4__layers1__schema1.pt \
        --output artifacts/oracle_dashboard/e0g-qwen-layer1.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Iterable

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import solution  # noqa: E402


ALL_SCALE_OFFSETS = tuple(range(-254, 255))
DEFAULT_ROLES = ("fc_gate", "fc_up", "v", "proj")


def _first_activation(folds: list[list[torch.Tensor]]) -> list[torch.Tensor]:
    """Flatten one cached calibration tensor per fold."""

    return [fold[0].to(torch.float32) for fold in folds]


def _layer_activation(
    folds: list[list[torch.Tensor]], layer_index: int
) -> list[torch.Tensor]:
    """Select one cached layer from each calibration window."""

    return [fold[layer_index].to(torch.float32) for fold in folds]


def _block_losses(
    dense: torch.Tensor,
    params: dict[str, torch.Tensor],
    gram64: torch.Tensor | None,
) -> torch.Tensor:
    rows, channels = map(int, dense.shape)
    blocks = channels // 64
    reference = dense.reshape(rows, blocks, 64).to(torch.float32)
    quantized = solution._dequantize_hif4(params).reshape(rows, blocks, 64)
    error = quantized - reference
    if gram64 is None:
        return error.square().sum(dim=-1)
    return torch.einsum("rbi,bij,rbj->rb", error, gram64, error)


def _code_tensor(params: dict[str, torch.Tensor]) -> torch.Tensor:
    scale = params["scale_factor"].squeeze(-1).squeeze(-1).squeeze(-1)
    return solution._e6m2_encode_nearest(scale).to(torch.int64)


@torch.no_grad()
def _score_side(
    dense: torch.Tensor,
    *,
    gram64: torch.Tensor | None,
    max_rows: int,
    top_blocks: int,
) -> dict[str, Any]:
    dense = dense.detach().to(torch.float32)
    rows, channels = map(int, dense.shape)
    if channels % 64:
        return {"skipped": "channels_not_divisible_by_64", "shape": [rows, channels]}

    sample_rows = min(rows, max(1, int(max_rows)))
    sample = dense[:sample_rows]
    baseline = solution._encode_rows(
        sample, solution._BASE_OFFSETS, gram64=gram64
    )
    oracle = solution._encode_rows(sample, ALL_SCALE_OFFSETS, gram64=gram64)
    baseline_loss = _block_losses(sample, baseline, gram64)
    oracle_loss = _block_losses(sample, oracle, gram64)
    gap = (baseline_loss - oracle_loss).clamp_min(0.0)
    relative = gap / baseline_loss.clamp_min(1.0e-12)
    flat = relative.flatten()
    count = min(flat.numel(), max(1, int(top_blocks)))
    top_values, top_indices = torch.topk(flat, k=count)
    baseline_codes = _code_tensor(baseline)
    oracle_codes = _code_tensor(oracle)
    top_rows = (top_indices // relative.shape[1]).tolist()
    top_blocks_local = (top_indices % relative.shape[1]).tolist()
    top_code_pairs = [
        {
            "row": int(row),
            "block": int(block),
            "relative_gap": float(value),
            "baseline_code": int(baseline_codes[row, block]),
            "oracle_code": int(oracle_codes[row, block]),
            "code_distance": int(
                abs(int(oracle_codes[row, block]) - int(baseline_codes[row, block]))
            ),
        }
        for value, row, block in zip(
            top_values.tolist(), top_rows, top_blocks_local
        )
    ]
    total_baseline = float(baseline_loss.sum())
    total_oracle = float(oracle_loss.sum())
    return {
        "shape": [rows, channels],
        "sample_rows": sample_rows,
        "sample_blocks": int(relative.numel()),
        "improved_blocks": int((gap > 1.0e-8).sum()),
        "mean_block_relative_gap": float(relative.mean()),
        "total_relative_gap": (total_baseline - total_oracle)
        / max(total_baseline, 1.0e-12),
        "max_block_relative_gap": float(relative.max()),
        "top_blocks": top_code_pairs,
    }


@torch.no_grad()
def run_oracle(
    cache_path: Path,
    roles: Iterable[str],
    *,
    max_rows: int,
    top_blocks: int,
    layer_index: int = 0,
) -> dict[str, Any]:
    cache = torch.load(cache_path, map_location="cpu", weights_only=False)
    layer_weights = cache["weights"][layer_index]
    calibration = cache["calibration_activations"]
    result: dict[str, Any] = {
        "schema": 1,
        "cache": str(cache_path),
        "model": cache.get("tokenizer_name", "unknown"),
        "layers": cache.get("layers"),
        "layer_index": int(layer_index),
        "max_rows": int(max_rows),
        "top_blocks": int(top_blocks),
        "scale_candidates": {"local_offsets": list(solution._BASE_OFFSETS), "oracle_codes": 255},
        "roles": {},
    }

    for role in roles:
        weight = layer_weights[role].to(torch.float32)
        activation_folds = _layer_activation(calibration[role], layer_index)
        balance, seed, block_size = solution._choose_boat(weight, activation_folds)
        weight_t = solution._apply_boat_rotation(
            weight * balance.reshape(1, -1), seed, block_size
        )
        activation_t = solution._apply_boat_rotation(
            activation_folds[0] / balance.reshape(1, -1), seed, block_size
        )
        gram64 = solution._gram64(weight_t)
        result["roles"][role] = {
            "boat": {
                "seed": int(seed),
                "block_size": int(block_size),
                "balance_min": float(balance.min()),
                "balance_max": float(balance.max()),
            },
            "weight_plain_mse": _score_side(
                weight_t, gram64=None, max_rows=max_rows, top_blocks=top_blocks
            ),
            "activation_gram": _score_side(
                activation_t, gram64=gram64, max_rows=max_rows, top_blocks=top_blocks
            ),
        }
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cache",
        type=Path,
        default=ROOT
        / "artifacts"
        / "real_model_suite"
        / "cache"
        / "qwen2.5-0.5b__seq128__calib2__test4__layers1__schema1.pt",
    )
    parser.add_argument("--roles", nargs="+", default=list(DEFAULT_ROLES))
    parser.add_argument("--max-rows", type=int, default=32)
    parser.add_argument("--top-blocks", type=int, default=32)
    parser.add_argument("--layer-index", type=int, default=0)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    started = time.perf_counter()
    result = run_oracle(
        args.cache,
        args.roles,
        max_rows=args.max_rows,
        top_blocks=args.top_blocks,
        layer_index=args.layer_index,
    )
    result["elapsed_seconds"] = time.perf_counter() - started
    encoded = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output is None:
        print(encoded, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
        print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
