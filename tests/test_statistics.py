from __future__ import annotations

import pytest

from hif4_system.models import CaseResult, TimingResult
from hif4_system.statistics import compare, decide, summarize


def _case(index: int, score: float, seed_id: str = "seed-00") -> CaseResult:
    return CaseResult(seed_id, "linear", "balanced", index, False, "fp32", 1.0, 1.0 - score, score)


def test_bootstrap_summary_is_repeatable_without_numpy() -> None:
    cases = tuple(_case(index, 0.1 * index) for index in range(1, 6))

    left = summarize(cases, rounds=200, seed=17)
    right = summarize(cases, rounds=200, seed=17)

    assert left == right
    assert left.case_count == 5
    assert left.negative_rate == 0.0


def test_summary_bootstrap_clusters_correlated_cases_by_seed() -> None:
    compact = (_case(0, 0.0, "seed-00"), _case(0, 1.0, "seed-01"))
    repeated = tuple(
        _case(index, score, seed_id)
        for seed_id, score in (("seed-00", 0.0), ("seed-01", 1.0))
        for index in range(50)
    )

    compact_ci = summarize(compact, rounds=500, seed=23).bootstrap_mean_ci95
    repeated_ci = summarize(repeated, rounds=500, seed=23).bootstrap_mean_ci95



def test_pairing_rejects_mismatched_case_sets() -> None:
    candidate = (_case(0, 0.1), _case(1, 0.2))
    champion = (_case(0, 0.1),)

    with pytest.raises(ValueError, match="paired case keys"):
        compare(candidate, champion, rounds=100, seed=9)


def test_decision_requires_authoritative_cpu_timing() -> None:
    summary = summarize((_case(0, 0.2),), rounds=50, seed=1)
    decision = decide(
        summary,
        None,
        None,
        TimingResult(1.0, 1.0),
        None,
        {"min_mean_score": 0.0, "max_negative_rate": 0.1, "max_catastrophic_rate": 0.02, "min_worst_decile_mean": -0.05, "min_candidate_delta": 0.002, "max_negative_rate_delta": 0.0, "max_runtime_ratio": 1.2},
        authoritative_timing=False,
    )

    assert decision.promote is False
    assert decision.checks["authoritative_timing"] is False
