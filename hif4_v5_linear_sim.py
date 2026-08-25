#!/usr/bin/env python3
"""Paired NumPy ablation for Linear data-driven HiF4 scale candidates."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

import hif4_numpy_sim as base


BASE_POLICY: Dict[str, Any] = {
    "weight_offsets": (-1, 1, 2), "weight_ranks": (),
    "weight_margin": 0.0, "weight_ratio": 0.20,
    "activation_offsets": (-1, 2), "activation_ranks": (),
    "activation_margin": 0.02, "activation_ratio": 0.10,
}


def policy(**updates: Any) -> Dict[str, Any]:
    value = dict(BASE_POLICY)
    value.update(updates)
    return value


POLICIES: Dict[str, Dict[str, Any]] = {
    "v3_current": policy(),
    "weight_r2": policy(weight_ranks=(2,), weight_margin=0.005),
    "weight_m2": policy(weight_offsets=(-2, -1, 1, 2)),
    "weight_m3": policy(weight_offsets=(-3, -2, -1, 1, 2)),
    "weight_p3": policy(weight_offsets=(-1, 1, 2, 3)),
    "weight_wide": policy(weight_offsets=(-2, -1, 1, 2, 3)),
    "weight_ratio10": policy(weight_ratio=0.10),
    "weight_ratio30": policy(weight_ratio=0.30),
    "activation_r24": policy(activation_ranks=(2, 4)),
    "activation_p1": policy(activation_offsets=(-1, 1, 2)),
    "activation_m2": policy(activation_offsets=(-2, -1, 2)),
    "activation_wide": policy(activation_offsets=(-2, -1, 1, 2)),
    "activation_p3": policy(activation_offsets=(-1, 1, 2, 3)),
    "activation_ratio05": policy(activation_ratio=0.05),
    "activation_ratio15": policy(activation_ratio=0.15),
    "activation_ratio20": policy(activation_ratio=0.20),
    "activation_ratio25": policy(activation_ratio=0.25),
    "activation_ratio30": policy(activation_ratio=0.30),
    "activation_ratio40": policy(activation_ratio=0.40),
    "weight_ratio40": policy(weight_ratio=0.40),
    "combo_a20_w30": policy(activation_ratio=0.20, weight_ratio=0.30),
    "combo_a15_w30": policy(activation_ratio=0.15, weight_ratio=0.30),
    "combo_a20_p3": policy(activation_ratio=0.20,
                            activation_offsets=(-1, 1, 2, 3)),
    "combo_a20_w30_p3": policy(activation_ratio=0.20, weight_ratio=0.30,
                                activation_offsets=(-1, 1, 2, 3)),
    "combo_a10_w30_p3": policy(activation_ratio=0.10, weight_ratio=0.30,
                                activation_offsets=(-1, 1, 2, 3)),
    "combo_a15_w30_p3": policy(activation_ratio=0.15, weight_ratio=0.30,
                                activation_offsets=(-1, 1, 2, 3)),
    "combo_a15_w30_p1": policy(activation_ratio=0.15, weight_ratio=0.30,
                                activation_offsets=(-1, 1, 2)),
    "combo_a20_w30_p1": policy(activation_ratio=0.20, weight_ratio=0.30,
                                activation_offsets=(-1, 1, 2)),
    "combo_a20_w30_wide": policy(activation_ratio=0.20, weight_ratio=0.30,
                                  activation_offsets=(-2, -1, 1, 2)),
    "combo_a20_w30_wp3": policy(activation_ratio=0.20, weight_ratio=0.30,
                                 weight_offsets=(-1, 1, 2, 3)),
    "both_r2": policy(weight_ranks=(2,), weight_margin=0.005,
                       activation_ranks=(2,)),
}


def calibrate_variants(
    weight: np.ndarray, calib: Sequence[np.ndarray]
) -> Dict[str, Tuple[np.ndarray, Dict[str, Any]]]:
    # Transform selection is independent of the final difficult-block search.
    _, transform_state = base.calibrate_linear(weight, calib, refine_weight=False)
    order = transform_state["order"]
    d_inv = transform_state["d_inv"]
    d = 1.0 / d_inv
    transformed_weight = (weight * d[None, :])[:, order]
    activation_second = np.mean(
        np.concatenate([np.square(x) for x in calib], axis=0), axis=0
    )
    h_x = (activation_second / np.square(d))[order]
    results: Dict[str, Tuple[np.ndarray, Dict[str, Any]]] = {}
    for name, current in POLICIES.items():
        weight_hat = base.hif4_quantize_reconstruct(
            transformed_weight,
            importance=h_x,
            offsets=current["weight_offsets"],
            top_ranks=current["weight_ranks"],
            accept_margin=current["weight_margin"],
            max_refine_ratio=current["weight_ratio"],
            max_refine_blocks=65_536,
        )
        importance = base.normalize_importance(
            np.sum(np.square(weight_hat), axis=0), weight.shape[1]
        )
        results[name] = (weight_hat, {
            "d_inv": d_inv, "order": order, "importance": importance,
            "activation_offsets": current["activation_offsets"],
            "activation_ranks": current["activation_ranks"],
            "activation_margin": current["activation_margin"],
            "activation_ratio": current["activation_ratio"],
        })
    return results


def dynamic_activation(
    x: np.ndarray, state: Dict[str, Any]
) -> np.ndarray:
    transformed = (x * state["d_inv"][None, :])[:, state["order"]]
    return base.hif4_quantize_reconstruct(
        transformed,
        importance=state["importance"],
        offsets=state["activation_offsets"],
        top_ranks=state["activation_ranks"],
        accept_margin=state["activation_margin"],
        max_refine_ratio=state["activation_ratio"],
        max_refine_blocks=32_768,
    )


def simulate_seed(seed: int) -> List[Dict[str, Any]]:
    rng = np.random.default_rng(seed)
    rows: List[Dict[str, Any]] = []
    out_features, channels, tokens = 96, 256, 32
    for scenario in ("balanced", "hierarchy", "outlier", "heavy_tail"):
        ap = base.profile(rng, channels, scenario)
        wp = np.clip(1.0 / ap, 0.05, 20.0)
        weight = base.nvfp4_dequantize(base.nvfp4_quantize(base.random_matrix(
            rng, out_features, wp, scenario == "heavy_tail"
        )))
        calibration = [base.nvfp4_dequantize(base.nvfp4_quantize(base.random_matrix(
            rng, tokens, ap, scenario == "heavy_tail"
        ))) for _ in range(2)]
        tests = [base.nvfp4_dequantize(base.nvfp4_quantize(base.random_matrix(
            rng, tokens, ap, scenario == "heavy_tail"
        ))) for _ in range(2)]
        standard_weight = base.hif4_quantize_reconstruct(weight)
        variants = calibrate_variants(weight, calibration)
        for test_index, activation in enumerate(tests):
            reference = activation @ weight.T
            standard_activation = base.hif4_quantize_reconstruct(activation)
            standard = standard_activation @ standard_weight.T
            mse_std = float(np.mean(np.square(reference - standard)))
            for name, (weight_hat, state) in variants.items():
                activation_hat = dynamic_activation(activation, state)
                output = activation_hat @ weight_hat.T
                mse_player = float(np.mean(np.square(reference - output)))
                rows.append({
                    "seed": seed, "kind": "linear", "variant": name,
                    "scenario": scenario, "test": test_index,
                    "mse_std": mse_std, "mse_player": mse_player,
                    "score": (mse_std - mse_player) / max(mse_std, 1.0e-30),
                })
    return rows


def summarize(rows: Sequence[Dict[str, Any]], elapsed: float) -> Dict[str, Any]:
    result: Dict[str, Any] = {"elapsed_seconds": elapsed, "variants": {}}
    for name in POLICIES:
        selected = [x for x in rows if x["variant"] == name]
        scores = np.asarray([x["score"] for x in selected], dtype=np.float64)
        tail = np.sort(scores)[:max(1, int(math.ceil(0.10 * len(scores))))]
        scenarios: Dict[str, Any] = {}
        for scenario in sorted({x["scenario"] for x in selected}):
            values = np.asarray([
                x["score"] for x in selected if x["scenario"] == scenario
            ], dtype=np.float64)
            scenarios[scenario] = {
                "mean": float(np.mean(values)), "min": float(np.min(values)),
                "negative": int(np.sum(values < 0.0)), "cases": len(values),
            }
        result["variants"][name] = {
            "cases": len(scores), "mean": float(np.mean(scores)),
            "median": float(np.median(scores)),
            "p05": float(np.quantile(scores, 0.05)), "min": float(np.min(scores)),
            "negative": int(np.sum(scores < 0.0)),
            "catastrophic": int(np.sum(scores < -0.10)),
            "worst_decile_mean": float(np.mean(tail)), "by_scenario": scenarios,
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", default="17,29,43,71,101")
    parser.add_argument("--output", default="hif4_v5_linear_result.json")
    args = parser.parse_args()
    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]
    start = time.perf_counter()
    rows: List[Dict[str, Any]] = []
    for seed in seeds:
        rows.extend(simulate_seed(seed))
    elapsed = time.perf_counter() - start
    report = {
        "metadata": {"backend": "numpy-linear-mirror", "seeds": seeds,
                     "note": "Trend only; validate with PyTorch."},
        "summary": summarize(rows, elapsed), "cases": rows,
    }
    Path(args.output).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    main()
