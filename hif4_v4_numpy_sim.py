#!/usr/bin/env python3
"""NumPy mirror and ablation runner for the v4 HiF4 Attention policy.

This file focuses on the v4 additions that are absent from hif4_numpy_sim.py:
real Attention-output gating, Jacobian importance, data-driven scale candidates,
and calibration-selected QK/QKV refinement.  It compares several safety gates
on identical calibration/test samples.  It is a trend simulator, not a
replacement for the PyTorch/official scorer.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

import hif4_numpy_sim as base


EPS = 1.0e-12


def transform_pair(
    q: np.ndarray, k: np.ndarray, d_kv: np.ndarray,
    q_heads: int, kv_heads: int, dim: int,
    q_order: np.ndarray, k_order: np.ndarray, center_mode: int,
) -> Tuple[np.ndarray, np.ndarray]:
    group = q_heads // kv_heads
    d_q = np.repeat(d_kv, group, axis=0).reshape(-1)
    d_k = (1.0 / d_kv).reshape(-1)
    qt = (q * d_q[None, :])[:, q_order]
    kt = (base.center_k(k, kv_heads, dim, center_mode) * d_k[None, :])[:, k_order]
    return qt.astype(np.float32), kt.astype(np.float32)


def quantize_v4(
    x: np.ndarray, importance: np.ndarray | None = None,
    ratio: float = 0.0, margin: float = 0.0,
) -> np.ndarray:
    if ratio <= 0.0:
        return base.hif4_quantize_reconstruct(x)
    return base.hif4_quantize_reconstruct(
        x, importance=importance, offsets=(-1, 2), top_ranks=(2,),
        accept_margin=margin, max_refine_ratio=ratio,
        max_refine_blocks=24_576,
    )


def true_metrics(
    q_samples: Sequence[np.ndarray], k_samples: Sequence[np.ndarray],
    v_samples: Sequence[np.ndarray], d_kv: np.ndarray,
    q_heads: int, kv_heads: int, dim: int,
    q_order: np.ndarray, k_order: np.ndarray, center_mode: int,
    q_importance: np.ndarray | None = None,
    k_importance: np.ndarray | None = None,
    refine_mode: int = 0,
) -> Tuple[float, Tuple[float, ...]]:
    values: List[float] = []
    for q, k, v in zip(q_samples, k_samples, v_samples):
        qt, kt = transform_pair(
            q, k, d_kv, q_heads, kv_heads, dim,
            q_order, k_order, center_mode,
        )
        qh = quantize_v4(qt, q_importance, 0.08 if refine_mode >= 1 else 0.0, 0.03)
        kh = quantize_v4(kt, k_importance, 0.12 if refine_mode >= 1 else 0.0, 0.03)
        vh = quantize_v4(v, None, 0.10 if refine_mode >= 2 else 0.0, 0.01)
        for causal in (False, True):
            reference = base.attention_output(q, k, v, q_heads, kv_heads, dim, causal)
            candidate = base.attention_output(qh, kh, vh, q_heads, kv_heads, dim, causal)
            loss = float(np.mean(np.square(reference - candidate)))
            energy = float(np.mean(np.square(reference)))
            values.append(loss / (energy + EPS))
    return float(np.mean(values)), tuple(values)


def jacobian_importance(
    q_samples: Sequence[np.ndarray], k_samples: Sequence[np.ndarray],
    v_samples: Sequence[np.ndarray], d_kv: np.ndarray,
    q_heads: int, kv_heads: int, dim: int,
    q_order: np.ndarray, k_order: np.ndarray, center_mode: int,
) -> Tuple[np.ndarray, np.ndarray]:
    group = q_heads // kv_heads
    qi = np.zeros((q_heads, dim), dtype=np.float64)
    ki = np.zeros((q_heads, dim), dtype=np.float64)
    count = 0
    for q, k, v in zip(q_samples, k_samples, v_samples):
        qt, kt = transform_pair(
            q, k, d_kv, q_heads, kv_heads, dim,
            q_order, k_order, center_mode,
        )
        seq = q.shape[0]
        qh = qt.reshape(seq, q_heads, dim).transpose(1, 0, 2)
        kh = np.repeat(kt.reshape(seq, kv_heads, dim), group, axis=1).transpose(1, 0, 2)
        vh = np.repeat(v.reshape(seq, kv_heads, dim), group, axis=1).transpose(1, 0, 2)
        raw = np.matmul(qh, np.swapaxes(kh, -1, -2)) / math.sqrt(float(dim))
        for causal in (False, True):
            logits = raw.copy()
            if causal:
                mask = np.triu(np.ones((seq, seq), dtype=bool), k=1)
                logits = np.where(mask[None, :, :], -1.0e30, logits)
            probabilities = base.softmax(logits)
            output = np.matmul(probabilities, vh)
            v_norm = np.sum(np.square(vh), axis=-1)
            o_norm = np.sum(np.square(output), axis=-1)
            cross = np.matmul(output, np.swapaxes(vh, -1, -2))
            distance = np.maximum(o_norm[..., None] + v_norm[:, None, :] - 2.0 * cross, 0.0)
            sensitivity = np.square(probabilities) * distance
            qi += np.einsum("hij,hjd->hd", sensitivity, np.square(kh)) / float(dim)
            ki += np.einsum("hij,hid->hd", sensitivity, np.square(qh)) / float(dim)
            count += 1
    if count:
        qi /= count
        ki /= count
    ki_kv = np.mean(ki.reshape(kv_heads, group, dim), axis=1)
    return (
        base.normalize_importance(qi.reshape(-1).astype(np.float32), q_heads * dim),
        base.normalize_importance(ki_kv.reshape(-1).astype(np.float32), kv_heads * dim),
    )


def robust_safe(
    candidate: Tuple[float, Tuple[float, ...]],
    baseline: Tuple[float, Tuple[float, ...]],
    mean_margin: float, worst_tolerance: float,
    mask_consensus: bool,
) -> bool:
    if not base.candidate_safe(candidate, baseline, mean_margin, worst_tolerance):
        return False
    if not mask_consensus:
        return True
    # Cases are ordered [sample0 noncausal, sample0 causal, ...].  Require
    # both mask domains to improve; this prevents one mask hiding the other.
    for mask_index in (0, 1):
        cur = candidate[1][mask_index::2]
        ref = baseline[1][mask_index::2]
        if float(np.mean(cur)) > float(np.mean(ref)) * (1.0 - 0.002):
            return False
    return True


def calibrate_attention_v4(
    q_samples: Sequence[np.ndarray], k_samples: Sequence[np.ndarray],
    v_samples: Sequence[np.ndarray], q_heads: int, kv_heads: int, dim: int,
    mean_margin: float = 0.005, worst_tolerance: float = 0.005,
    mask_consensus: bool = False,
    flat_threshold: float = 0.0,
    moderate_threshold: float = 0.0,
    shrink_power: float = 1.0,
    flat_refine: bool = False,
    flat_refine_cap: int = 2,
    true_transform_gate: bool = True,
) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    proxy_q, proxy_k, _ = base.calibrate_attention(
        q_samples, k_samples, q_heads, kv_heads, dim
    )
    group = q_heads // kv_heads
    identity_d = np.ones((kv_heads, dim), dtype=np.float32)
    local_identity = np.broadcast_to(np.arange(dim), (kv_heads, dim))
    k_identity = base.flatten_head_order(local_identity)
    q_identity = base.flatten_head_order(np.repeat(local_identity, group, axis=0))
    proxy_d = proxy_q["multiplier"].reshape(q_heads, dim).reshape(
        kv_heads, group, dim
    )[:, 0, :]
    proxy_q_order = proxy_q["order"]
    proxy_k_order = proxy_k["order"]
    proxy_center = int(proxy_k["center"])

    q_peak = np.sqrt(np.mean(np.stack([
        np.square(np.max(np.abs(q.reshape(-1, q_heads, dim)), axis=0))
        for q in q_samples
    ]), axis=0))
    k_peak = np.sqrt(np.mean(np.stack([
        np.square(np.max(np.abs(k.reshape(-1, kv_heads, dim)), axis=0))
        for k in k_samples
    ]), axis=0))
    q_peak_kv = np.max(q_peak.reshape(kv_heads, group, dim), axis=1)
    q_log = np.log2(np.maximum(q_peak_kv, EPS))
    k_log = np.log2(np.maximum(k_peak, EPS))
    pressure = np.maximum(
        q_log - np.median(q_log, axis=-1, keepdims=True),
        k_log - np.median(k_log, axis=-1, keepdims=True),
    )
    pressure_span = float(np.quantile(pressure, 0.95) - np.quantile(pressure, 0.05))
    flat_profile = float(flat_threshold) > 0.0 and pressure_span < float(flat_threshold)
    if (not flat_profile and float(moderate_threshold) > 0.0
            and pressure_span < float(moderate_threshold)):
        proxy_d = np.power(proxy_d, float(shrink_power)).astype(np.float32)

    baseline = true_metrics(
        q_samples, k_samples, v_samples, identity_d,
        q_heads, kv_heads, dim, q_identity, k_identity, 0,
    )
    best = baseline
    selected = (identity_d, q_identity, k_identity, 0)
    if not flat_profile and not true_transform_gate:
        selected = (proxy_d, proxy_q_order, proxy_k_order, proxy_center)
    candidates = (
        (proxy_d, q_identity, k_identity, proxy_center),
        (proxy_d, proxy_q_order, proxy_k_order, proxy_center),
    )
    seen: set[Tuple[int, bytes, bytes]] = set()
    gated_candidates = candidates if true_transform_gate else ()
    for d, qo, ko, center in (() if flat_profile else gated_candidates):
        key = (center, np.asarray(qo).tobytes(), np.asarray(ko).tobytes())
        if key in seen:
            continue
        seen.add(key)
        metrics = true_metrics(
            q_samples, k_samples, v_samples, d,
            q_heads, kv_heads, dim, qo, ko, center,
        )
        if metrics[0] < best[0] and robust_safe(
            metrics, baseline, mean_margin, worst_tolerance, mask_consensus
        ):
            best, selected = metrics, (d, qo, ko, center)

    d, q_order, k_order, center = selected
    q_stack = np.concatenate(q_samples, axis=0).reshape(-1, q_heads, dim)
    centered_k = np.concatenate([
        base.center_k(k, kv_heads, dim, center) for k in k_samples
    ], axis=0).reshape(-1, kv_heads, dim)
    q_second = np.mean(np.square(q_stack), axis=0)
    k_second = np.mean(np.square(centered_k), axis=0)
    q_second_kv = np.mean(q_second.reshape(kv_heads, group, dim), axis=1)
    d_q = np.repeat(d, group, axis=0)
    d_k = 1.0 / d
    moment_q = base.normalize_importance(
        np.repeat(k_second * np.square(d_k), group, axis=0).reshape(-1)[q_order],
        q_heads * dim,
    )
    moment_k = base.normalize_importance(
        (q_second_kv * np.square(d)).reshape(-1)[k_order], kv_heads * dim,
    )
    if not flat_profile or flat_refine:
        jq, jk = jacobian_importance(
            q_samples, k_samples, v_samples, d,
            q_heads, kv_heads, dim, q_order, k_order, center,
        )
        q_importance = base.normalize_importance(
            0.25 * moment_q + 0.75 * jq, q_heads * dim
        )
        k_importance = base.normalize_importance(
            0.25 * moment_k + 0.75 * jk, kv_heads * dim
        )
    else:
        q_importance, k_importance = moment_q, moment_k

    refine_baseline = true_metrics(
        q_samples, k_samples, v_samples, d, q_heads, kv_heads, dim,
        q_order, k_order, center, q_importance, k_importance, 0,
    )
    refine_best = refine_baseline
    refine_mode = 0
    refine_modes = (1,) if flat_profile and int(flat_refine_cap) == 1 else (1, 2)
    for mode in (() if flat_profile and not flat_refine else refine_modes):
        metrics = true_metrics(
            q_samples, k_samples, v_samples, d, q_heads, kv_heads, dim,
            q_order, k_order, center, q_importance, k_importance, mode,
        )
        if metrics[0] < refine_best[0] and robust_safe(
            metrics, refine_baseline, mean_margin, worst_tolerance, mask_consensus
        ):
            refine_best, refine_mode = metrics, mode
    return (
        {"multiplier": d_q.reshape(-1), "order": q_order,
         "importance": q_importance, "refine_mode": refine_mode,
         "flat_profile": flat_profile},
        {"multiplier": d_k.reshape(-1), "order": k_order, "center": center,
         "importance": k_importance, "refine_mode": refine_mode,
         "flat_profile": flat_profile},
        {"refine_mode": refine_mode, "flat_profile": flat_profile},
    )


def dynamic_q_v4(x: np.ndarray, state: Dict[str, Any]) -> np.ndarray:
    transformed = (x * state["multiplier"][None, :])[:, state["order"]]
    return quantize_v4(
        transformed, state["importance"],
        0.08 if state["refine_mode"] >= 1 else 0.0, 0.03,
    )


def dynamic_k_v4(x: np.ndarray, state: Dict[str, Any], heads: int, dim: int) -> np.ndarray:
    centered = base.center_k(x, heads, dim, state["center"])
    transformed = (centered * state["multiplier"][None, :])[:, state["order"]]
    return quantize_v4(
        transformed, state["importance"],
        0.12 if state["refine_mode"] >= 1 else 0.0, 0.03,
    )


def dynamic_v_v4(x: np.ndarray, state: Dict[str, Any]) -> np.ndarray:
    return quantize_v4(x, None, 0.10 if state["refine_mode"] >= 2 else 0.0, 0.01)


GATES = {
    "v4_current": (0.005, 0.005, False, 0.0, 0.0, 1.0, False, 2, True),
    "v4_flat075": (0.005, 0.005, False, 0.75, 0.0, 1.0, False, 2, True),
    "v4_flat075_qkv": (0.005, 0.005, False, 0.75, 0.0, 1.0, True, 2, True),
    "v5_hybrid_proxy": (0.005, 0.005, False, 0.75, 0.0, 1.0, True, 2, False),
}


def simulate_attention_seed(seed: int) -> List[Dict[str, Any]]:
    rng = np.random.default_rng(seed)
    cases: List[Dict[str, Any]] = []
    q_heads, kv_heads, dim, seq = 4, 2, 64, 32
    group = q_heads // kv_heads
    for scenario in ("balanced", "qk_imbalance", "k_shift", "heavy_tail"):
        base_mode = "hierarchy" if scenario == "qk_imbalance" else (
            "heavy_tail" if scenario == "heavy_tail" else "balanced")
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
        states = {
            name: calibrate_attention_v4(
                [x[0] for x in calibration], [x[1] for x in calibration],
                [x[2] for x in calibration], q_heads, kv_heads, dim, *gate,
            ) for name, gate in GATES.items()
        }
        v3_states = base.calibrate_attention(
            [x[0] for x in calibration], [x[1] for x in calibration],
            q_heads, kv_heads, dim,
        )
        states["v3_current"] = v3_states
        flat_states = states["v4_flat075_qkv"]
        use_v3 = not bool(flat_states[0]["flat_profile"])
        source_states = v3_states if use_v3 else flat_states
        hybrid_states = tuple(dict(item) for item in source_states)
        hybrid_states[0]["_use_v3"] = use_v3
        states["v5_hybrid_v3"] = hybrid_states
        for test_index in range(2):
            q, k, v = sample()
            q_std = base.hif4_quantize_reconstruct(q)
            k_std = base.hif4_quantize_reconstruct(k)
            v_std = base.hif4_quantize_reconstruct(v)
            for causal in (False, True):
                reference = base.attention_output(q, k, v, q_heads, kv_heads, dim, causal)
                standard = base.attention_output(q_std, k_std, v_std,
                                                 q_heads, kv_heads, dim, causal)
                mse_std = float(np.mean(np.square(reference - standard)))
                for name, (qs, ks, vs) in states.items():
                    if name == "v3_current" or bool(qs.get("_use_v3", False)):
                        qh = base.dynamic_q(q, qs, refine=True)
                        kh = base.dynamic_k(k, ks, kv_heads, dim, refine=True)
                        vh = base.dynamic_v(v, refine=True, state=vs)
                    else:
                        qh = dynamic_q_v4(q, qs)
                        kh = dynamic_k_v4(k, ks, kv_heads, dim)
                        vh = dynamic_v_v4(v, vs)
                    output = base.attention_output(qh, kh, vh,
                                                   q_heads, kv_heads, dim, causal)
                    mse_player = float(np.mean(np.square(reference - output)))
                    cases.append({
                        "seed": seed, "kind": "attention", "variant": name,
                        "scenario": scenario, "test": test_index, "causal": causal,
                        "refine_mode": int(qs.get("refine_mode", 2)),
                        "transform": int(ks["center"] != 0 or
                                         not np.array_equal(ks["order"], np.arange(kv_heads * dim)) or
                                         not np.allclose(ks["multiplier"], 1.0)),
                        "mse_std": mse_std, "mse_player": mse_player,
                        "score": (mse_std - mse_player) / max(mse_std, 1.0e-30),
                    })
    return cases


def summarize(cases: Sequence[Dict[str, Any]], elapsed: float) -> Dict[str, Any]:
    result: Dict[str, Any] = {"elapsed_seconds": elapsed, "variants": {}}
    for variant in ("v3_current", *GATES, "v5_hybrid_v3"):
        rows = [x for x in cases if x["variant"] == variant]
        scores = np.asarray([x["score"] for x in rows], dtype=np.float64)
        tail = np.sort(scores)[:max(1, int(math.ceil(0.1 * len(scores))))]
        by_scenario: Dict[str, Any] = {}
        for scenario in sorted({x["scenario"] for x in rows}):
            selected = [x for x in rows if x["scenario"] == scenario]
            values = np.asarray([x["score"] for x in selected])
            by_scenario[scenario] = {
                "mean": float(np.mean(values)), "min": float(np.min(values)),
                "negative": int(np.sum(values < 0.0)), "cases": len(values),
            }
        result["variants"][variant] = {
            "cases": len(rows), "mean": float(np.mean(scores)),
            "median": float(np.median(scores)), "p05": float(np.quantile(scores, 0.05)),
            "min": float(np.min(scores)), "negative": int(np.sum(scores < 0.0)),
            "negative_rate": float(np.mean(scores < 0.0)),
            "catastrophic": int(np.sum(scores < -0.1)),
            "worst_decile_mean": float(np.mean(tail)),
            "transform_rate": float(np.mean([x["transform"] for x in rows])),
            "refine_rate": float(np.mean([x["refine_mode"] > 0 for x in rows])),
            "by_scenario": by_scenario,
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", default="17,29,43,71,101")
    parser.add_argument("--output", default="hif4_v4_numpy_result.json")
    args = parser.parse_args()
    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]
    start = time.perf_counter()
    cases: List[Dict[str, Any]] = []
    for seed in seeds:
        cases.extend(simulate_attention_seed(seed))
    elapsed = time.perf_counter() - start
    report = {
        "metadata": {"backend": "numpy-v4-mirror", "seeds": seeds,
                     "note": "Trend only; validate selected policy with PyTorch."},
        "summary": summarize(cases, elapsed), "cases": cases,
    }
    output = Path(args.output)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report["summary"], indent=2))
    print("result=%s" % output.resolve())


if __name__ == "__main__":
    main()
