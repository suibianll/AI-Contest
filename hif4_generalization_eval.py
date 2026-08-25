#!/usr/bin/env python3
"""HiF4 algorithm generalization evaluation environment.

The evaluator has two backends:

* torch: authoritative path.  It imports ``hif4_score_eval.py``, loads actual
  competition solutions and evaluates operator output MSE.
* numpy: lightweight protocol smoke test for machines without PyTorch.  It
  reuses ``hif4_numpy_sim.py`` and must not be used as the final score.

The holdout split is generated from a campaign-local secret and a monotonically
increasing attempt number.  Seeds are never stored in reports.  Candidate and
incumbent are evaluated on exactly the same data, which makes paired deltas
useful while reducing repeated tuning against a fixed public test set.
"""

from __future__ import annotations

import argparse
import hashlib
import hmac
import importlib
import json
import math
import os
import secrets
import statistics
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple


SCHEMA_VERSION = 1
DEFAULT_DEV_SEEDS = (101, 211, 307)
DEFAULT_THRESHOLDS: Dict[str, float] = {
    "min_mean_score": 0.0,
    "max_negative_rate": 0.10,
    "max_catastrophic_rate": 0.02,
    "min_worst_decile_mean": -0.05,
    "min_candidate_delta": 0.002,
    "max_negative_rate_delta": 0.0,
    "max_runtime_ratio": 1.20,
}
TIERS: Dict[str, Dict[str, int]] = {
    "smoke": {"dev_seeds": 1, "holdout_seeds": 1, "calib": 1, "tests": 1},
    "standard": {"dev_seeds": 3, "holdout_seeds": 5, "calib": 2, "tests": 2},
    "soak": {"dev_seeds": 8, "holdout_seeds": 12, "calib": 3, "tests": 3},
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def percentile(values: Sequence[float], q: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(float(x) for x in values)
    position = (len(ordered) - 1) * q
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def mean_or_nan(values: Sequence[float]) -> float:
    return statistics.fmean(values) if values else float("nan")


def bootstrap_mean_ci(
    values: Sequence[float], seed: int = 20260825, rounds: int = 2000
) -> Tuple[float, float]:
    if not values:
        return float("nan"), float("nan")
    if len(values) == 1:
        return float(values[0]), float(values[0])
    try:
        import numpy as np
    except ImportError:
        return float("nan"), float("nan")
    array = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(array), size=(rounds, len(array)))
    samples = np.mean(array[indices], axis=1)
    return float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))


def aggregate_cases(cases: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    scores = [float(row["score"]) for row in cases]
    if not scores:
        return {"case_count": 0}
    tail_count = max(1, int(math.ceil(0.10 * len(scores))))
    worst = sorted(scores)[:tail_count]
    ci_low, ci_high = bootstrap_mean_ci(scores)
    ratios = [float(row.get("mse_player", 0.0)) /
              max(float(row.get("mse_std", 0.0)), 1.0e-30) for row in cases]
    result: Dict[str, Any] = {
        "case_count": len(scores),
        "score_sum": float(sum(scores)),
        "score_mean": mean_or_nan(scores),
        "score_median": float(statistics.median(scores)),
        "score_p05": percentile(scores, 0.05),
        "score_min": min(scores),
        "worst_decile_mean": mean_or_nan(worst),
        "negative_cases": sum(value < 0.0 for value in scores),
        "negative_rate": sum(value < 0.0 for value in scores) / len(scores),
        "catastrophic_cases": sum(value < -0.10 for value in scores),
        "catastrophic_rate": sum(value < -0.10 for value in scores) / len(scores),
        "mse_ratio_mean": mean_or_nan(ratios),
        "bootstrap_mean_ci95": [ci_low, ci_high],
    }
    by_dimension: Dict[str, Any] = {}
    for dimension in ("kind", "scenario", "causal", "compute_dtype"):
        groups: Dict[str, List[Mapping[str, Any]]] = {}
        for row in cases:
            key = str(row.get(dimension, "unknown"))
            groups.setdefault(key, []).append(row)
        by_dimension[dimension] = {
            key: aggregate_cases_without_groups(rows) for key, rows in sorted(groups.items())
        }
    result["groups"] = by_dimension
    return result


def aggregate_cases_without_groups(cases: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    scores = [float(row["score"]) for row in cases]
    tail_count = max(1, int(math.ceil(0.10 * len(scores))))
    return {
        "cases": len(scores),
        "mean": mean_or_nan(scores),
        "p05": percentile(scores, 0.05),
        "min": min(scores),
        "worst_decile_mean": mean_or_nan(sorted(scores)[:tail_count]),
        "negative_rate": sum(value < 0.0 for value in scores) / len(scores),
    }


def paired_comparison(
    candidate: Sequence[Mapping[str, Any]],
    incumbent: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    def key(row: Mapping[str, Any]) -> Tuple[Any, ...]:
        return (
            row.get("seed_id"), row.get("kind"), row.get("scenario"),
            row.get("test"), row.get("causal"), row.get("compute_dtype"),
        )

    right = {key(row): row for row in incumbent}
    deltas: List[float] = []
    wins = ties = losses = 0
    missing = 0
    for row in candidate:
        other = right.get(key(row))
        if other is None:
            missing += 1
            continue
        delta = float(row["score"]) - float(other["score"])
        deltas.append(delta)
        if delta > 1.0e-12:
            wins += 1
        elif delta < -1.0e-12:
            losses += 1
        else:
            ties += 1
    ci_low, ci_high = bootstrap_mean_ci(deltas, seed=20260826)
    return {
        "paired_cases": len(deltas),
        "missing_cases": missing,
        "mean_score_delta": mean_or_nan(deltas),
        "median_score_delta": float(statistics.median(deltas)) if deltas else float("nan"),
        "p05_score_delta": percentile(deltas, 0.05),
        "min_score_delta": min(deltas) if deltas else float("nan"),
        "win_tie_loss": [wins, ties, losses],
        "win_rate": wins / len(deltas) if deltas else float("nan"),
        "bootstrap_delta_ci95": [ci_low, ci_high],
    }


@dataclass(frozen=True)
class Campaign:
    directory: Path
    manifest_path: Path
    secret_path: Path
    manifest: Dict[str, Any]
    secret: bytes


def load_or_create_campaign(directory: Path, max_holdout_uses: int) -> Campaign:
    directory.mkdir(parents=True, exist_ok=True)
    manifest_path = directory / "campaign.json"
    secret_path = directory / ".holdout_secret"
    if manifest_path.exists() != secret_path.exists():
        raise RuntimeError("campaign.json and .holdout_secret must exist together")
    if not manifest_path.exists():
        secret = secrets.token_bytes(32)
        secret_path.write_bytes(secret)
        try:
            os.chmod(secret_path, 0o600)
        except OSError:
            pass
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "campaign_id": secrets.token_hex(8),
            "created_at": utc_now(),
            "dev_seeds": list(DEFAULT_DEV_SEEDS),
            "holdout_uses": 0,
            "max_holdout_uses": int(max_holdout_uses),
            "runs": [],
        }
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    else:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        secret = secret_path.read_bytes()
    return Campaign(directory, manifest_path, secret_path, manifest, secret)


def save_campaign(campaign: Campaign) -> None:
    campaign.manifest_path.write_text(
        json.dumps(campaign.manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def derive_holdout_seeds(campaign: Campaign, attempt: int, count: int) -> List[int]:
    values: List[int] = []
    for index in range(count):
        message = "%s:%d:%d" % (campaign.manifest["campaign_id"], attempt, index)
        digest = hmac.new(campaign.secret, message.encode("utf-8"), hashlib.sha256).digest()
        value = int.from_bytes(digest[:8], "big") % 2_000_000_000 + 1
        values.append(value)
    return values


def seed_commitment(campaign: Campaign, seeds: Sequence[int]) -> str:
    payload = canonical_json(list(seeds)).encode("utf-8")
    return hmac.new(campaign.secret, payload, hashlib.sha256).hexdigest()


def select_seeds(
    campaign: Campaign, split: str, tier: str
) -> Tuple[List[int], Optional[int], str]:
    spec = TIERS[tier]
    if split == "dev":
        required = spec["dev_seeds"]
        seeds = list(campaign.manifest["dev_seeds"])
        while len(seeds) < required:
            seeds.append(401 + 97 * len(seeds))
        return seeds[:required], None, hashlib.sha256(canonical_json(seeds[:required]).encode()).hexdigest()
    used = int(campaign.manifest["holdout_uses"])
    maximum = int(campaign.manifest["max_holdout_uses"])
    if used >= maximum:
        raise RuntimeError("holdout budget exhausted (%d/%d)" % (used, maximum))
    attempt = used + 1
    seeds = derive_holdout_seeds(campaign, attempt, spec["holdout_seeds"])
    return seeds, attempt, seed_commitment(campaign, seeds)


def torch_available() -> bool:
    try:
        import torch  # noqa: F401
        return True
    except ImportError:
        return False


def load_config(path: Optional[str]) -> Dict[str, Any]:
    result: Dict[str, Any] = {"thresholds": dict(DEFAULT_THRESHOLDS)}
    if path:
        supplied = json.loads(Path(path).read_text(encoding="utf-8"))
        result.update({key: value for key, value in supplied.items() if key != "thresholds"})
        result["thresholds"].update(supplied.get("thresholds", {}))
    return result


def _annotate(
    cases: Sequence[Dict[str, Any]], groups: Sequence[Mapping[str, Any]],
    seed_id: str, causal: bool, compute_dtype: str,
) -> List[Dict[str, Any]]:
    annotated: List[Dict[str, Any]] = []
    for row in cases:
        item = dict(row)
        item["scenario"] = str(groups[int(row["group"])].get("name", "unknown"))
        item["seed_id"] = seed_id
        item["causal"] = bool(causal)
        item["compute_dtype"] = compute_dtype
        annotated.append(item)
    return annotated


def build_torch_suite(seed: int, tier: str) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Build matched and shifted synthetic operator suites using PyTorch."""
    import torch
    scorer = importlib.import_module("hif4_score_eval")
    spec = TIERS[tier]
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    if tier == "smoke":
        channels, out_features, tokens, seq = 128, 48, 17, 24
        linear_modes = ("balanced", "outlier", "calib_test_shift")
        attention_modes = ("balanced", "k_shift", "saturated_logits")
    elif tier == "standard":
        channels, out_features, tokens, seq = 256, 128, 49, 48
        linear_modes = ("balanced", "hierarchy", "outlier", "heavy_tail",
                        "calib_test_shift", "correlated", "sparse")
        attention_modes = ("balanced", "qk_imbalance", "k_shift", "heavy_tail",
                           "saturated_logits", "v_outlier", "qk_correlation")
    else:
        channels, out_features, tokens, seq = 512, 256, 97, 80
        linear_modes = ("balanced", "hierarchy", "outlier", "heavy_tail",
                        "calib_test_shift", "correlated", "sparse", "mean_shift")
        attention_modes = ("balanced", "qk_imbalance", "k_shift", "heavy_tail",
                           "saturated_logits", "v_outlier", "qk_correlation", "mean_shift")

    def matrix(rows: int, profile: Any, heavy: bool = False,
               mean: Optional[Any] = None, correlated: bool = False,
               sparse: bool = False) -> Any:
        value = scorer._random_matrix(rows, profile, generator, heavy, mean)
        if correlated:
            grouped = value.reshape(rows, -1, 64)
            shared = grouped.mean(dim=-1, keepdim=True)
            value = (0.55 * grouped + 0.45 * shared).reshape(rows, -1)
        if sparse:
            mask = torch.rand(value.shape, generator=generator) < 0.70
            value = torch.where(mask, torch.zeros_like(value), value)
        return value

    linear_groups: List[Dict[str, Any]] = []
    for mode in linear_modes:
        profile_mode = mode if mode in ("balanced", "hierarchy", "outlier", "heavy_tail") else "hierarchy"
        calib_profile = scorer._profile(channels, profile_mode, generator)
        test_profile = calib_profile
        if mode == "calib_test_shift":
            drift = torch.logspace(-0.65, 0.65, channels)
            drift = drift[torch.randperm(channels, generator=generator)]
            test_profile = calib_profile * drift
        weight_profile = calib_profile.reciprocal().clamp(0.05, 20.0)
        mean = (0.75 * torch.randn(channels, generator=generator)
                if mode == "mean_shift" else None)
        weight = matrix(out_features, weight_profile, mode == "heavy_tail",
                        correlated=mode == "correlated", sparse=mode == "sparse")
        calib = [scorer.quantize_to_nvfp4(matrix(
            tokens, calib_profile, mode == "heavy_tail", mean,
            correlated=mode == "correlated", sparse=mode == "sparse"
        )) for _ in range(spec["calib"])]
        tests = [scorer.quantize_to_nvfp4(matrix(
            tokens + (index % 2) * 16, test_profile, mode in ("heavy_tail", "calib_test_shift"),
            mean, correlated=mode == "correlated", sparse=mode == "sparse"
        )) for index in range(spec["tests"])]
        linear_groups.append({
            "name": mode, "weight": scorer.quantize_to_nvfp4(weight),
            "calib_activation_list": calib, "test_activation_list": tests,
        })

    attention_groups: List[Dict[str, Any]] = []
    head_specs = [(4, 2, 64)]
    if tier != "smoke":
        head_specs += [(8, 2, 64), (8, 8, 64)]
    if tier == "soak":
        head_specs += [(4, 1, 128)]
    for mode_index, mode in enumerate(attention_modes):
        q_heads, kv_heads, head_dim = head_specs[mode_index % len(head_specs)]
        group_size = q_heads // kv_heads
        kv_channels = kv_heads * head_dim
        q_channels = q_heads * head_dim
        base_mode = "hierarchy" if mode in ("qk_imbalance", "qk_correlation") else (
            "heavy_tail" if mode in ("heavy_tail", "v_outlier") else "balanced")
        kp = scorer._profile(kv_channels, base_mode, generator)
        qp_kv = kp.reciprocal().clamp(0.05, 20.0)
        qp = qp_kv.reshape(kv_heads, head_dim).repeat_interleave(group_size, dim=0).reshape(q_channels)
        shift = (2.5 * torch.randn(kv_channels, generator=generator)
                 if mode in ("k_shift", "mean_shift") else None)

        def attention_sample(index: int, calibration: bool) -> Dict[str, Any]:
            length = seq if calibration else seq + (index % 2) * 16
            q = matrix(length, qp, mode == "heavy_tail")
            k = matrix(length, kp, mode == "heavy_tail", shift,
                       correlated=mode == "qk_correlation")
            if mode == "qk_correlation":
                shared = torch.randn(length, kv_heads, head_dim, generator=generator)
                k = 0.65 * k.reshape(length, kv_heads, head_dim) + 0.35 * shared
                q_shared = shared.repeat_interleave(group_size, dim=1)
                q = 0.65 * q.reshape(length, q_heads, head_dim) + 0.35 * q_shared
                q, k = q.reshape(length, q_channels), k.reshape(length, kv_channels)
            if mode == "saturated_logits":
                factor = 2.5 if calibration else 3.25
                q, k = q * factor, k * factor
            vp = torch.ones(kv_channels)
            v = matrix(length, vp, mode in ("heavy_tail", "v_outlier"))
            if mode == "v_outlier" and not calibration:
                v = v * torch.where(
                    torch.rand(v.shape, generator=generator) < 0.01,
                    torch.full_like(v, 10.0), torch.ones_like(v),
                )
            return {"q": scorer.quantize_to_nvfp4(q),
                    "k": scorer.quantize_to_nvfp4(k),
                    "v": scorer.quantize_to_nvfp4(v)}

        attention_groups.append({
            "name": "%s_h%d_kv%d_d%d" % (mode, q_heads, kv_heads, head_dim),
            "q_num_heads": q_heads, "kv_num_heads": kv_heads, "head_dim": head_dim,
            "calib": [attention_sample(i, True) for i in range(spec["calib"])],
            "test": [attention_sample(i, False) for i in range(spec["tests"])],
        })
    return linear_groups, attention_groups


def evaluate_torch_solution(
    solution_path: Path, seeds: Sequence[int], tier: str,
    causal_modes: Sequence[bool], compute_dtypes: Sequence[str],
) -> Tuple[List[Dict[str, Any]], Dict[str, float]]:
    scorer = importlib.import_module("hif4_score_eval")
    solution = scorer.load_solution(str(solution_path))
    cases: List[Dict[str, Any]] = []
    quant_seconds = wall_seconds = 0.0
    for seed_index, seed in enumerate(seeds):
        linear_groups, attention_groups = build_torch_suite(seed, tier)
        seed_id = "seed-%02d" % seed_index
        for dtype in compute_dtypes:
            start = time.perf_counter()
            linear_rows, linear_time = scorer.evaluate_linear_groups(
                linear_groups, solution, dtype
            )
            wall_seconds += time.perf_counter() - start
            quant_seconds += float(linear_time["player_quant_seconds"])
            cases.extend(_annotate(linear_rows, linear_groups, seed_id, False, dtype))
            for causal in causal_modes:
                start = time.perf_counter()
                attention_rows, attention_time = scorer.evaluate_attention_groups(
                    attention_groups, solution, dtype, causal
                )
                wall_seconds += time.perf_counter() - start
                quant_seconds += float(attention_time["player_quant_seconds"])
                cases.extend(_annotate(attention_rows, attention_groups, seed_id, causal, dtype))
    return cases, {"player_quant_seconds": quant_seconds, "evaluation_wall_seconds": wall_seconds}


def evaluate_numpy_proxy(
    seeds: Sequence[int], causal_modes: Sequence[bool], variant: str,
) -> Tuple[List[Dict[str, Any]], Dict[str, float]]:
    """Run the existing NumPy mirror as a protocol and metric smoke test."""
    import hif4_numpy_sim as mirror
    start = time.perf_counter()
    selected: List[Dict[str, Any]] = []
    for seed_index, seed in enumerate(seeds):
        for mask_index, causal in enumerate(causal_modes):
            rows = mirror.simulate_seed(seed, causal)
            for row in rows:
                if row["variant"] != variant:
                    continue
                # Linear does not depend on the Attention mask.  Keep it once
                # when evaluating both causal and non-causal Attention.
                if row["kind"] == "linear" and mask_index > 0:
                    continue
                item = dict(row)
                item["seed_id"] = "seed-%02d" % seed_index
                item["causal"] = causal if item["kind"] == "attention" else False
                item["compute_dtype"] = "fp32-numpy-mirror"
                selected.append(item)
    elapsed = time.perf_counter() - start
    return selected, {"player_quant_seconds": elapsed, "evaluation_wall_seconds": elapsed}


def evaluate_numpy_v4_proxy(
    seeds: Sequence[int], variant: str,
) -> Tuple[List[Dict[str, Any]], Dict[str, float]]:
    """Run the v4 Attention mirror without exposing raw holdout seeds."""
    import hif4_v4_numpy_sim as mirror
    start = time.perf_counter()
    selected: List[Dict[str, Any]] = []
    for seed_index, seed in enumerate(seeds):
        for row in mirror.simulate_attention_seed(seed):
            if row["variant"] != variant:
                continue
            item = dict(row)
            item.pop("seed", None)
            item["seed_id"] = "seed-%02d" % seed_index
            item["compute_dtype"] = "fp32-numpy-v4-mirror"
            selected.append(item)
    elapsed = time.perf_counter() - start
    return selected, {"player_quant_seconds": elapsed, "evaluation_wall_seconds": elapsed}


def evaluate_numpy_linear_proxy(
    seeds: Sequence[int], variant: str,
) -> Tuple[List[Dict[str, Any]], Dict[str, float]]:
    """Run the Linear search mirror without exposing raw holdout seeds."""
    import hif4_v5_linear_sim as mirror
    start = time.perf_counter()
    selected: List[Dict[str, Any]] = []
    for seed_index, seed in enumerate(seeds):
        for row in mirror.simulate_seed(seed):
            if row["variant"] != variant:
                continue
            item = dict(row)
            item.pop("seed", None)
            item["seed_id"] = "seed-%02d" % seed_index
            item["causal"] = False
            item["compute_dtype"] = "fp32-numpy-linear-mirror"
            selected.append(item)
    elapsed = time.perf_counter() - start
    return selected, {"player_quant_seconds": elapsed, "evaluation_wall_seconds": elapsed}


def promotion_decision(
    candidate_summary: Mapping[str, Any], incumbent_summary: Optional[Mapping[str, Any]],
    comparison: Optional[Mapping[str, Any]], candidate_time: float,
    incumbent_time: Optional[float], thresholds: Mapping[str, float], backend: str,
) -> Dict[str, Any]:
    checks = {
        "mean_score": float(candidate_summary["score_mean"]) >= float(thresholds["min_mean_score"]),
        "negative_rate": float(candidate_summary["negative_rate"]) <= float(thresholds["max_negative_rate"]),
        "catastrophic_rate": float(candidate_summary["catastrophic_rate"]) <= float(thresholds["max_catastrophic_rate"]),
        "tail_score": float(candidate_summary["worst_decile_mean"]) >= float(thresholds["min_worst_decile_mean"]),
    }
    if comparison is not None and incumbent_summary is not None:
        checks["paired_delta"] = float(comparison["mean_score_delta"]) >= float(thresholds["min_candidate_delta"])
        checks["negative_rate_delta"] = (
            float(candidate_summary["negative_rate"]) - float(incumbent_summary["negative_rate"])
            <= float(thresholds["max_negative_rate_delta"])
        )
    if incumbent_time is not None and incumbent_time > 0.0:
        checks["runtime"] = candidate_time / incumbent_time <= float(thresholds["max_runtime_ratio"])
    if backend != "torch":
        checks["authoritative_backend"] = False
    return {
        "promote": all(checks.values()),
        "checks": checks,
        "note": ("PASS means statistically eligible, not guaranteed leaderboard gain."
                 if backend == "torch" else
                 "NumPy is a trend-only smoke test; promotion is always blocked."),
    }


def parse_bool_modes(value: str) -> List[bool]:
    if value == "both":
        return [False, True]
    return [value == "causal"]


def run(args: argparse.Namespace) -> Dict[str, Any]:
    campaign = load_or_create_campaign(Path(args.campaign_dir), args.max_holdout_uses)
    seeds, attempt, commitment = select_seeds(campaign, args.split, args.tier)
    backend = args.backend
    if backend == "auto":
        backend = "torch" if torch_available() else "numpy"
    candidate_path = Path(args.candidate).resolve()
    if not candidate_path.is_file():
        raise FileNotFoundError(candidate_path)
    incumbent_path = Path(args.incumbent).resolve() if args.incumbent else None
    if incumbent_path is not None and not incumbent_path.is_file():
        raise FileNotFoundError(incumbent_path)
    causal_modes = parse_bool_modes(args.attention_mask)
    dtypes = [value.strip() for value in args.compute_dtypes.split(",") if value.strip()]
    config = load_config(args.config)

    if backend == "torch":
        candidate_cases, candidate_timing = evaluate_torch_solution(
            candidate_path, seeds, args.tier, causal_modes, dtypes
        )
        incumbent_cases = None
        incumbent_timing = None
        if incumbent_path is not None:
            incumbent_cases, incumbent_timing = evaluate_torch_solution(
                incumbent_path, seeds, args.tier, causal_modes, dtypes
            )
    elif backend == "numpy-v4":
        candidate_cases, candidate_timing = evaluate_numpy_v4_proxy(
            seeds, args.numpy_variant
        )
        incumbent_cases = None
        incumbent_timing = None
        if args.numpy_incumbent_variant:
            incumbent_cases, incumbent_timing = evaluate_numpy_v4_proxy(
                seeds, args.numpy_incumbent_variant
            )
    elif backend == "numpy-linear":
        candidate_cases, candidate_timing = evaluate_numpy_linear_proxy(
            seeds, args.numpy_variant
        )
        incumbent_cases = None
        incumbent_timing = None
        if args.numpy_incumbent_variant:
            incumbent_cases, incumbent_timing = evaluate_numpy_linear_proxy(
                seeds, args.numpy_incumbent_variant
            )
    else:
        candidate_cases, candidate_timing = evaluate_numpy_proxy(
            seeds, causal_modes, args.numpy_variant
        )
        incumbent_cases = None
        incumbent_timing = None
        if args.numpy_incumbent_variant:
            incumbent_cases, incumbent_timing = evaluate_numpy_proxy(
                seeds, causal_modes, args.numpy_incumbent_variant
            )

    candidate_summary = aggregate_cases(candidate_cases)
    incumbent_summary = aggregate_cases(incumbent_cases) if incumbent_cases else None
    comparison = paired_comparison(candidate_cases, incumbent_cases) if incumbent_cases else None
    decision = promotion_decision(
        candidate_summary, incumbent_summary, comparison,
        float(candidate_timing["player_quant_seconds"]),
        float(incumbent_timing["player_quant_seconds"]) if incumbent_timing else None,
        config["thresholds"], backend,
    )
    report: Dict[str, Any] = {
        "metadata": {
            "schema_version": SCHEMA_VERSION,
            "created_at": utc_now(),
            "campaign_id": campaign.manifest["campaign_id"],
            "split": args.split,
            "tier": args.tier,
            "backend": backend,
            "authoritative": backend == "torch",
            "seed_count": len(seeds),
            "seed_commitment": commitment,
            "holdout_attempt": attempt,
            "candidate": {
                "path": str(candidate_path),
                "sha256": sha256_file(candidate_path),
                "solution_code_executed": backend == "torch",
                "evaluated_object": (str(candidate_path) if backend == "torch" else
                                     ("hif4_v4_numpy_sim.py:%s" % args.numpy_variant
                                      if backend == "numpy-v4" else
                                      ("hif4_v5_linear_sim.py:%s" % args.numpy_variant
                                       if backend == "numpy-linear" else
                                       "hif4_numpy_sim.py:%s" % args.numpy_variant))),
            },
            "incumbent": ({
                "path": str(incumbent_path),
                "sha256": sha256_file(incumbent_path),
                "solution_code_executed": backend == "torch",
                "evaluated_object": (str(incumbent_path) if backend == "torch" else
                                     ("hif4_v4_numpy_sim.py:%s" % args.numpy_incumbent_variant
                                      if backend == "numpy-v4" else
                                      ("hif4_v5_linear_sim.py:%s" % args.numpy_incumbent_variant
                                       if backend == "numpy-linear" else
                                       "hif4_numpy_sim.py:%s" % args.numpy_incumbent_variant))),
            } if incumbent_path else None),
            "attention_mask": args.attention_mask,
            "compute_dtypes": (dtypes if backend == "torch" else
                               (["fp32-numpy-v4-mirror"] if backend == "numpy-v4"
                                else (["fp32-numpy-linear-mirror"]
                                      if backend == "numpy-linear"
                                      else ["fp32-numpy-mirror"]))),
        },
        "thresholds": config["thresholds"],
        "candidate": {"summary": candidate_summary, "timing": candidate_timing},
        "incumbent": ({"summary": incumbent_summary, "timing": incumbent_timing}
                      if incumbent_summary else None),
        "comparison": comparison,
        "decision": decision,
        "cases": candidate_cases if args.include_cases else None,
    }
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    if attempt is not None:
        campaign.manifest["holdout_uses"] = attempt
    campaign.manifest["runs"].append({
        "created_at": report["metadata"]["created_at"],
        "split": args.split, "tier": args.tier, "backend": backend,
        "seed_commitment": commitment, "holdout_attempt": attempt,
        "candidate_sha256": report["metadata"]["candidate"]["sha256"],
        "incumbent_sha256": (report["metadata"]["incumbent"] or {}).get("sha256"),
        "score_mean": candidate_summary["score_mean"],
        "negative_rate": candidate_summary["negative_rate"],
        "promote": decision["promote"], "report": str(output),
    })
    save_campaign(campaign)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", required=True,
                        help="candidate solution.py; hashed before evaluation")
    parser.add_argument("--incumbent", help="previous solution for paired comparison")
    parser.add_argument("--campaign-dir", default="hif4_eval_campaign")
    parser.add_argument("--split", choices=("dev", "holdout"), default="dev")
    parser.add_argument("--tier", choices=tuple(TIERS), default="standard")
    parser.add_argument("--backend", choices=("auto", "torch", "numpy", "numpy-v4", "numpy-linear"), default="auto")
    parser.add_argument("--attention-mask", choices=("both", "causal", "noncausal"), default="both")
    parser.add_argument("--compute-dtypes", default="fp32",
                        help="comma-separated: fp32,bf16")
    parser.add_argument("--config")
    parser.add_argument("--output", default="hif4_generalization_result.json")
    parser.add_argument("--include-cases", action="store_true")
    parser.add_argument("--max-holdout-uses", type=int, default=3)
    parser.add_argument("--numpy-variant", default="full_current")
    parser.add_argument("--numpy-incumbent-variant", default="transform_only")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    report = run(args)
    visible = {
        "metadata": report["metadata"],
        "candidate": report["candidate"],
        "incumbent": report["incumbent"],
        "comparison": report["comparison"],
        "decision": report["decision"],
    }
    print(json.dumps(visible, indent=2, ensure_ascii=False))
    print("result=%s" % Path(args.output).resolve())


if __name__ == "__main__":
    main()
