"""Measure recall of the analytical GALS-C scale candidate set.

This is an evaluator-side research prototype.  It deliberately does not modify
the submission path or write any activation state.  For each sampled 64-value
block it generates the critical scales

    s = |x_i| / (m * 2**e),  m in {1/4,...,7/4}, e in {0,1,2},

projects them to legal E6M2 codes and evaluates the resulting per-block
offsets with the same legal hierarchy solver used by ``solution.py``.  The
full 255-code oracle is used only as a recall reference.
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


ALL_OFFSETS = tuple(range(-254, 255))
MANTISSA = torch.as_tensor((0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75))
EXPONENT = torch.as_tensor((1.0, 2.0, 4.0))


def _block_loss(
    dense: torch.Tensor,
    params: dict[str, torch.Tensor],
    gram64: torch.Tensor | None,
) -> torch.Tensor:
    rows, channels = map(int, dense.shape)
    blocks = channels // 64
    error = (
        solution._dequantize_hif4(params).to(torch.float32) - dense.to(torch.float32)
    ).reshape(rows, blocks, 64)
    if gram64 is None:
        return error.square().sum(dim=-1)
    return torch.einsum("rbi,bij,rbj->rb", error, gram64, error)


def _candidate_offsets(dense: torch.Tensor) -> torch.Tensor:
    """Return [rows, blocks, K] analytical offsets, always including ±3."""
    rows, channels = map(int, dense.shape)
    blocks = channels // 64
    x = torch.nan_to_num(
        dense.to(torch.float32),
        nan=0.0,
        posinf=solution._E6M2_MAX * 7.0,
        neginf=-solution._E6M2_MAX * 7.0,
    ).reshape(rows, blocks, 64)
    absolute = x.abs()
    standard_code, _ = solution._standard_scale(absolute.amax(dim=-1))
    values = absolute[..., :, None, None] / (
        MANTISSA.to(x.device)[None, None, None, :, None]
        * EXPONENT.to(x.device)[None, None, None, None, :]
    )
    projected = solution._e6m2_encode_nearest(values)
    projected = torch.cat((projected - 1, projected, projected + 1), dim=-1)
    projected = projected.clamp(0, 254).reshape(rows, blocks, -1)
    offsets = projected - standard_code[..., None]
    base = torch.as_tensor(solution._BASE_OFFSETS, device=x.device).reshape(1, 1, -1)
    offsets = torch.cat((offsets, base.expand(rows, blocks, -1)), dim=-1)
    return torch.unique(offsets, dim=-1).to(torch.int64)


@torch.no_grad()
def _evaluate_candidate_set(
    dense: torch.Tensor,
    gram64: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return baseline, GALS-C, and all-code block losses."""
    dense = dense.to(torch.float32)
    baseline = solution._encode_rows(dense, solution._BASE_OFFSETS, gram64=gram64)
    baseline_loss = _block_loss(dense, baseline, gram64)
    oracle = solution._encode_rows(dense, ALL_OFFSETS, gram64=gram64)
    oracle_loss = _block_loss(dense, oracle, gram64)

    candidates = _candidate_offsets(dense)
    best_loss = baseline_loss.clone()
    for offset in torch.unique(candidates).tolist():
        params = solution._encode_rows(dense, (int(offset),), gram64=gram64)
        loss = _block_loss(dense, params, gram64)
        allowed = (candidates == int(offset)).any(dim=-1)
        best_loss = torch.where(allowed, torch.minimum(best_loss, loss), best_loss)
    return baseline_loss, best_loss, oracle_loss


@torch.no_grad()
def run(
    cache_path: Path,
    roles: Iterable[str],
    max_rows: int,
) -> dict[str, Any]:
    cache = torch.load(cache_path, map_location="cpu", weights_only=False)
    layer_weights = cache["weights"][0]
    calibration = cache["calibration_activations"]
    result: dict[str, Any] = {
        "schema": 1,
        "cache": str(cache_path),
        "model": cache.get("tokenizer_name", "unknown"),
        "max_rows": int(max_rows),
        "candidate_definition": {
            "mantissa": [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75],
            "effective_exponent": [0, 1, 2],
            "projected_neighbors": [-1, 0, 1],
            "incumbent_offsets": list(solution._BASE_OFFSETS),
        },
        "roles": {},
    }
    for role in roles:
        weight = layer_weights[role].to(torch.float32)
        samples = [fold[0].to(torch.float32) for fold in calibration[role]]
        balance, seed, block_size = solution._choose_boat(weight, samples)
        weight_t = solution._apply_boat_rotation(
            weight * balance.reshape(1, -1), seed, block_size
        )
        activation_t = solution._apply_boat_rotation(
            samples[0] / balance.reshape(1, -1), seed, block_size
        )
        gram64 = solution._gram64(weight_t)
        weight_sample = weight_t[: max(1, int(max_rows))]
        activation_sample = activation_t[: max(1, int(max_rows))]
        for name, dense, gram in (
            ("weight_plain_mse", weight_sample, None),
            ("activation_gram64", activation_sample, gram64),
        ):
            baseline, candidate, oracle = _evaluate_candidate_set(dense, gram)
            baseline_total = float(baseline.sum())
            oracle_total = float(oracle.sum())
            candidate_total = float(candidate.sum())
            oracle_gap = max(baseline_total - oracle_total, 0.0)
            result["roles"].setdefault(role, {})[name] = {
                "shape": list(dense.shape),
                "candidate_offsets": int(_candidate_offsets(dense).shape[-1]),
                "baseline_loss": baseline_total,
                "candidate_loss": candidate_total,
                "oracle_loss": oracle_total,
                "baseline_to_oracle_gap": oracle_gap / max(baseline_total, 1.0e-12),
                "candidate_to_oracle_recall": (
                    max(baseline_total - candidate_total, 0.0) / max(oracle_gap, 1.0e-12)
                ),
                "candidate_improved_blocks": int((candidate < baseline - 1.0e-8).sum()),
                "oracle_improved_blocks": int((oracle < baseline - 1.0e-8).sum()),
            }
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--roles", nargs="+", default=["fc_gate", "fc_up", "v", "proj"])
    parser.add_argument("--max-rows", type=int, default=32)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    started = time.perf_counter()
    result = run(args.cache, args.roles, args.max_rows)
    result["elapsed_seconds"] = time.perf_counter() - started
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.output} ({result['elapsed_seconds']:.2f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
