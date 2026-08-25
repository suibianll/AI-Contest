#!/usr/bin/env python3
"""Paired ablation for calibration-selected Q/K/V refinement budgets.

The transform and softmax-Jacobian importance come from the v5 RPSG policy.
This experiment changes only the online difficult-block ratios and selects them
with real short-window Attention output under causal and non-causal masks.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

import hif4_numpy_sim as base
import hif4_v4_numpy_sim as v5


Budget = Tuple[float, float, float]

FULL_BUDGETS: Tuple[Budget, ...] = (
    (0.00, 0.00, 0.00),
    (0.04, 0.08, 0.00),
    (0.08, 0.12, 0.00),
    (0.12, 0.16, 0.00),
    (0.04, 0.08, 0.05),
    (0.08, 0.12, 0.10),
    (0.12, 0.16, 0.10),
)

LOW_COST_BUDGETS: Tuple[Budget, ...] = (
    (0.00, 0.00, 0.00),
    (0.04, 0.08, 0.00),
    (0.08, 0.12, 0.00),
    (0.04, 0.08, 0.05),
    (0.08, 0.12, 0.10),
)

FACTORIZED_BUDGETS: Tuple[Budget, ...] = (
    (0.00, 0.00, 0.00),
    (0.04, 0.00, 0.00),
    (0.08, 0.00, 0.00),
    (0.00, 0.08, 0.00),
    (0.00, 0.12, 0.00),
    (0.00, 0.00, 0.05),
    (0.00, 0.00, 0.10),
    (0.04, 0.08, 0.00),
    (0.08, 0.12, 0.00),
    (0.04, 0.08, 0.05),
    (0.08, 0.12, 0.05),
    (0.08, 0.12, 0.10),
)

ESCALATION_BUDGETS: Tuple[Budget, ...] = (
    (0.04, 0.08, 0.05),
    (0.12, 0.16, 0.00),
    (0.12, 0.16, 0.10),
)

POLICIES: Dict[str, Dict[str, Any]] = {
    "v5_current": {"mode": "current"},
    "budget_full": {
        "budgets": FULL_BUDGETS, "mean_margin": 0.005,
        "worst_tolerance": 0.005, "mask_consensus": False,
        "cost_penalty": 0.0,
    },
    "budget_consensus": {
        "budgets": FULL_BUDGETS, "mean_margin": 0.005,
        "worst_tolerance": 0.005, "mask_consensus": True,
        "cost_penalty": 0.0,
    },
    "budget_risk0": {
        "budgets": FULL_BUDGETS, "mean_margin": 0.005,
        "worst_tolerance": 0.0, "mask_consensus": True,
        "cost_penalty": 0.0,
    },
    "budget_lowcost": {
        "budgets": LOW_COST_BUDGETS, "mean_margin": 0.005,
        "worst_tolerance": 0.005, "mask_consensus": True,
        "cost_penalty": 0.002,
    },
    "budget_vs_current": {
        "budgets": FULL_BUDGETS, "mean_margin": 0.010,
        "worst_tolerance": 0.0, "mask_consensus": True,
        "cost_penalty": 0.001, "baseline_current": True,
    },
    "budget_vs_current_strict": {
        "budgets": LOW_COST_BUDGETS, "mean_margin": 0.020,
        "worst_tolerance": 0.0, "mask_consensus": True,
        "cost_penalty": 0.002, "baseline_current": True,
    },
    "budget_nonflat": {
        "budgets": FULL_BUDGETS, "mean_margin": 0.010,
        "worst_tolerance": 0.0, "mask_consensus": True,
        "cost_penalty": 0.001, "baseline_current": True,
        "nonflat_only": True,
    },
    "budget_nonflat_strict": {
        "budgets": LOW_COST_BUDGETS, "mean_margin": 0.020,
        "worst_tolerance": 0.0, "mask_consensus": True,
        "cost_penalty": 0.002, "baseline_current": True,
        "nonflat_only": True,
    },
    "budget_factorized": {
        "budgets": FACTORIZED_BUDGETS, "mean_margin": 0.010,
        "worst_tolerance": 0.0, "mask_consensus": True,
        "cost_penalty": 0.001, "baseline_current": True,
    },
    "budget_factorized_strict": {
        "budgets": FACTORIZED_BUDGETS, "mean_margin": 0.020,
        "worst_tolerance": 0.0, "mask_consensus": True,
        "cost_penalty": 0.002, "baseline_current": True,
    },
    "budget_escalate": {
        "budgets": ESCALATION_BUDGETS, "mean_margin": 0.010,
        "worst_tolerance": 0.0, "mask_consensus": True,
        "cost_penalty": 0.001, "baseline_current": True,
    },
    "budget_high_qkv": {
        "budgets": ((0.12, 0.16, 0.10),), "mean_margin": 0.010,
        "worst_tolerance": 0.0, "mask_consensus": True,
        "cost_penalty": 0.001, "baseline_current": True,
    },
    "flat_guard_1pct": {
        "mode": "flat_guard", "mean_margin": 0.010,
        "worst_tolerance": 0.0, "mask_consensus": True,
    },
    "flat_guard_2pct": {
        "mode": "flat_guard", "mean_margin": 0.020,
        "worst_tolerance": 0.0, "mask_consensus": True,
    },
    "flat_guard_5pct": {
        "mode": "flat_guard", "mean_margin": 0.050,
        "worst_tolerance": 0.0, "mask_consensus": True,
    },
    "flat_cv_half": {
        "mode": "flat_guard", "mean_margin": 0.005,
        "worst_tolerance": 0.0, "mask_consensus": True,
        "subwindows": "half",
    },
    "flat_cv_interleave": {
        "mode": "flat_guard", "mean_margin": 0.005,
        "worst_tolerance": 0.0, "mask_consensus": True,
        "subwindows": "interleave",
    },
}


def clone_states(states: Sequence[Dict[str, Any]]):
    return tuple({k: (v.copy() if isinstance(v, np.ndarray) else v)
                  for k, v in state.items()} for state in states)


def quantize_budget(
    x: np.ndarray, importance: np.ndarray | None, ratio: float, margin: float
) -> np.ndarray:
    if ratio <= 0.0:
        return base.hif4_quantize_reconstruct(x)
    return base.hif4_quantize_reconstruct(
        x, importance=importance, offsets=(-1, 2), top_ranks=(2,),
        accept_margin=margin, max_refine_ratio=ratio,
        max_refine_blocks=24_576,
    )


def dynamic_budget(
    q: np.ndarray, k: np.ndarray, v: np.ndarray,
    states: Sequence[Dict[str, Any]], kv_heads: int, dim: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    qs, ks, vs = states
    qr, kr, vr = qs["budget"]
    qt = (q * qs["multiplier"][None, :])[:, qs["order"]]
    kc = base.center_k(k, kv_heads, dim, int(ks["center"]))
    kt = (kc * ks["multiplier"][None, :])[:, ks["order"]]
    return (
        quantize_budget(qt, qs["importance"], qr, 0.03),
        quantize_budget(kt, ks["importance"], kr, 0.03),
        quantize_budget(v, None, vr, 0.01),
    )


def budget_metrics(
    q_samples: Sequence[np.ndarray], k_samples: Sequence[np.ndarray],
    v_samples: Sequence[np.ndarray], states: Sequence[Dict[str, Any]],
    q_heads: int, kv_heads: int, dim: int,
) -> Tuple[float, Tuple[float, ...]]:
    values: List[float] = []
    for q, k, v in zip(q_samples, k_samples, v_samples):
        qh, kh, vh = dynamic_budget(q, k, v, states, kv_heads, dim)
        for causal in (False, True):
            reference = base.attention_output(
                q, k, v, q_heads, kv_heads, dim, causal
            )
            candidate = base.attention_output(
                qh, kh, vh, q_heads, kv_heads, dim, causal
            )
            loss = float(np.mean(np.square(reference - candidate)))
            energy = float(np.mean(np.square(reference)))
            values.append(loss / (energy + v5.EPS))
    return float(np.mean(values)), tuple(values)


def safe_budget(
    candidate: Tuple[float, Tuple[float, ...]],
    baseline: Tuple[float, Tuple[float, ...]],
    config: Dict[str, Any],
) -> bool:
    if not v5.robust_safe(
        candidate, baseline, float(config["mean_margin"]),
        float(config["worst_tolerance"]), bool(config["mask_consensus"]),
    ):
        return False
    return True


def choose_budget(
    q_samples: Sequence[np.ndarray], k_samples: Sequence[np.ndarray],
    v_samples: Sequence[np.ndarray], base_states: Sequence[Dict[str, Any]],
    q_heads: int, kv_heads: int, dim: int, config: Dict[str, Any],
):
    if (bool(config.get("nonflat_only", False)) and
            bool(base_states[0].get("flat_profile", False))):
        return current_states(base_states)
    if bool(config.get("baseline_current", False)):
        baseline_states = current_states(base_states)
    else:
        baseline_states = clone_states(base_states)
        baseline_states[0]["budget"] = (0.0, 0.0, 0.0)
        baseline_states[1]["budget"] = (0.0, 0.0, 0.0)
        baseline_states[2]["budget"] = (0.0, 0.0, 0.0)
    baseline = budget_metrics(
        q_samples, k_samples, v_samples, baseline_states,
        q_heads, kv_heads, dim,
    )
    best_states, best_metrics = baseline_states, baseline
    baseline_budget = tuple(float(x) for x in baseline_states[0]["budget"])
    best_objective = baseline[0] + float(config["cost_penalty"]) * sum(
        baseline_budget
    )
    for budget in config["budgets"]:
        if budget == (0.0, 0.0, 0.0):
            continue
        candidate_states = clone_states(base_states)
        for state in candidate_states:
            state["budget"] = tuple(float(x) for x in budget)
        current = budget_metrics(
            q_samples, k_samples, v_samples, candidate_states,
            q_heads, kv_heads, dim,
        )
        # Penalty is calibrated in normalized-loss units per unit budget.  It
        # breaks near-ties in favor of lower online work; safety is still
        # checked against the unpenalized real Attention loss.
        cost = sum(budget)
        objective = current[0] + float(config["cost_penalty"]) * cost
        if (objective < best_objective and
                safe_budget(current, baseline, config)):
            best_states, best_metrics = candidate_states, current
            best_objective = objective
    return best_states


def current_states(base_states: Sequence[Dict[str, Any]]):
    states = clone_states(base_states)
    mode = int(states[0].get("refine_mode", 0))
    budget = (
        0.08 if mode >= 1 else 0.0,
        0.12 if mode >= 1 else 0.0,
        0.10 if mode >= 2 else 0.0,
    )
    for state in states:
        state["budget"] = budget
    return states


def guard_flat_refinement(
    q_samples: Sequence[np.ndarray], k_samples: Sequence[np.ndarray],
    v_samples: Sequence[np.ndarray], base_states: Sequence[Dict[str, Any]],
    q_heads: int, kv_heads: int, dim: int, config: Dict[str, Any],
):
    current = current_states(base_states)
    if (not bool(current[0].get("flat_profile", False)) or
            sum(current[0]["budget"]) <= 0.0):
        return current
    fallback = clone_states(base_states)
    for state in fallback:
        state["budget"] = (0.0, 0.0, 0.0)
    baseline_metrics = budget_metrics(
        q_samples, k_samples, v_samples, fallback,
        q_heads, kv_heads, dim,
    )
    current_metrics = budget_metrics(
        q_samples, k_samples, v_samples, current,
        q_heads, kv_heads, dim,
    )
    if v5.robust_safe(
        current_metrics, baseline_metrics, float(config["mean_margin"]),
        float(config["worst_tolerance"]), bool(config["mask_consensus"]),
    ):
        selected = current
    else:
        return fallback
    split_mode = config.get("subwindows")
    if not split_mode:
        return selected
    q_split: List[np.ndarray] = []
    k_split: List[np.ndarray] = []
    v_split: List[np.ndarray] = []
    for q, k, v in zip(q_samples, k_samples, v_samples):
        if split_mode == "interleave":
            slices = (slice(0, None, 2), slice(1, None, 2))
        else:
            middle = max(1, q.shape[0] // 2)
            slices = (slice(0, middle), slice(middle, None))
        for current_slice in slices:
            if q[current_slice].shape[0] < 2:
                continue
            q_split.append(q[current_slice])
            k_split.append(k[current_slice])
            v_split.append(v[current_slice])
    split_baseline = budget_metrics(
        q_split, k_split, v_split, fallback, q_heads, kv_heads, dim
    )
    split_current = budget_metrics(
        q_split, k_split, v_split, current, q_heads, kv_heads, dim
    )
    if v5.robust_safe(
        split_current, split_baseline, float(config["mean_margin"]),
        float(config["worst_tolerance"]), bool(config["mask_consensus"]),
    ):
        return current
    return fallback


def simulate_seed(
    seed: int, variant_names: Optional[Sequence[str]] = None
) -> List[Dict[str, Any]]:
    rng = np.random.default_rng(seed)
    cases: List[Dict[str, Any]] = []
    q_heads, kv_heads, dim, seq = 4, 2, 64, 32
    group = q_heads // kv_heads
    for scenario in ("balanced", "qk_imbalance", "k_shift", "heavy_tail"):
        base_mode = "hierarchy" if scenario == "qk_imbalance" else (
            "heavy_tail" if scenario == "heavy_tail" else "balanced"
        )
        kp = base.profile(rng, kv_heads * dim, base_mode)
        qp = np.repeat(
            np.clip(1.0 / kp, 0.05, 20.0).reshape(kv_heads, dim),
            group, axis=0,
        ).reshape(q_heads * dim)
        shift = (3.0 * rng.standard_normal(kv_heads * dim, dtype=np.float32)
                 if scenario == "k_shift" else None)

        def sample() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
            q = base.nvfp4_dequantize(base.nvfp4_quantize(base.random_matrix(
                rng, seq, qp, scenario == "heavy_tail")))
            k = base.nvfp4_dequantize(base.nvfp4_quantize(base.random_matrix(
                rng, seq, kp, scenario == "heavy_tail", shift)))
            v = base.nvfp4_dequantize(base.nvfp4_quantize(base.random_matrix(
                rng, seq, np.ones(kv_heads * dim, dtype=np.float32),
                scenario == "heavy_tail")))
            return q, k, v

        calibration = [sample() for _ in range(2)]
        q_cal = [x[0] for x in calibration]
        k_cal = [x[1] for x in calibration]
        v_cal = [x[2] for x in calibration]
        base_states = v5.calibrate_attention_v4(
            q_cal, k_cal, v_cal, q_heads, kv_heads, dim,
            0.005, 0.005, False, 0.75, 0.0, 1.0, True, 2, True, False,
        )
        variants = {"v5_current": current_states(base_states)}
        enabled = set(variant_names or POLICIES.keys())
        for name, config in POLICIES.items():
            if name == "v5_current":
                continue
            if name not in enabled:
                continue
            if config.get("mode") == "flat_guard":
                variants[name] = guard_flat_refinement(
                    q_cal, k_cal, v_cal, base_states,
                    q_heads, kv_heads, dim, config,
                )
            else:
                variants[name] = choose_budget(
                    q_cal, k_cal, v_cal, base_states,
                    q_heads, kv_heads, dim, config,
                )
        for test_index in range(2):
            q, k, v = sample()
            q_std = base.hif4_quantize_reconstruct(q)
            k_std = base.hif4_quantize_reconstruct(k)
            v_std = base.hif4_quantize_reconstruct(v)
            for causal in (False, True):
                reference = base.attention_output(
                    q, k, v, q_heads, kv_heads, dim, causal
                )
                standard = base.attention_output(
                    q_std, k_std, v_std, q_heads, kv_heads, dim, causal
                )
                mse_std = float(np.mean(np.square(reference - standard)))
                for name, states in variants.items():
                    qh, kh, vh = dynamic_budget(q, k, v, states, kv_heads, dim)
                    output = base.attention_output(
                        qh, kh, vh, q_heads, kv_heads, dim, causal
                    )
                    mse_player = float(np.mean(np.square(reference - output)))
                    budget = tuple(float(x) for x in states[0]["budget"])
                    cases.append({
                        "seed": seed, "variant": name, "scenario": scenario,
                        "test": test_index, "causal": causal,
                        "budget": budget, "budget_sum": float(sum(budget)),
                        "center": int(states[1].get("center", 0)),
                        "flat_profile": bool(states[0].get("flat_profile", False)),
                        "mse_std": mse_std, "mse_player": mse_player,
                        "score": (mse_std - mse_player) / max(mse_std, 1.0e-30),
                    })
    return cases


def summarize(cases: Sequence[Dict[str, Any]], elapsed: float) -> Dict[str, Any]:
    result: Dict[str, Any] = {"elapsed_seconds": elapsed, "variants": {}}
    baseline = np.asarray([
        x["score"] for x in cases if x["variant"] == "v5_current"
    ])
    active_names = [
        name for name in POLICIES
        if any(x["variant"] == name for x in cases)
    ]
    for name in active_names:
        rows = [x for x in cases if x["variant"] == name]
        scores = np.asarray([x["score"] for x in rows], dtype=np.float64)
        delta = scores - baseline
        tail = np.sort(scores)[:max(1, int(math.ceil(0.1 * len(scores))))]
        result["variants"][name] = {
            "cases": len(rows), "mean": float(np.mean(scores)),
            "paired_delta": float(np.mean(delta)),
            "wins": int(np.sum(delta > 1.0e-12)),
            "ties": int(np.sum(np.abs(delta) <= 1.0e-12)),
            "losses": int(np.sum(delta < -1.0e-12)),
            "negative": int(np.sum(scores < 0.0)),
            "catastrophic": int(np.sum(scores < -0.10)),
            "p05": float(np.quantile(scores, 0.05)),
            "min": float(np.min(scores)),
            "worst_decile_mean": float(np.mean(tail)),
            "mean_budget": float(np.mean([x["budget_sum"] for x in rows])),
            "active_rate": float(np.mean([x["budget_sum"] > 0 for x in rows])),
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", default="17,29,43,71,101")
    parser.add_argument("--output", default="hif4_v9_attention_budget.json")
    parser.add_argument(
        "--variants", default="",
        help="Comma-separated candidate variants; v5_current is always included.",
    )
    args = parser.parse_args()
    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]
    variant_names = [x for x in args.variants.split(",") if x.strip()] or None
    start = time.perf_counter()
    cases: List[Dict[str, Any]] = []
    for seed in seeds:
        cases.extend(simulate_seed(seed, variant_names))
    elapsed = time.perf_counter() - start
    report = {
        "metadata": {"backend": "numpy-attention-mirror", "seeds": seeds,
                     "note": "Trend only; validate with PyTorch."},
        "summary": summarize(cases, elapsed), "cases": cases,
    }
    Path(args.output).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    main()
