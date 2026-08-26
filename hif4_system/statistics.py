from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import torch

from .models import CaseResult, TimingResult


def _percentile(values: Sequence[float], quantile: float) -> float:
    if not values:
        return float("nan")
    data = torch.tensor(values, dtype=torch.float64)
    q = torch.tensor(quantile, dtype=data.dtype)
    return float(torch.quantile(data, q).item())


def bootstrap_mean_ci(values: Sequence[float], rounds: int, seed: int) -> tuple[float, float]:
    if not values:
        return float("nan"), float("nan")
    if len(values) == 1:
        return float(values[0]), float(values[0])
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    data = torch.tensor(values, dtype=torch.float64)
    indices = torch.randint(len(values), (rounds, len(values)), generator=generator)
    means = data[indices].mean(dim=1)
    q = torch.tensor([0.025, 0.975], dtype=data.dtype)
    interval = torch.quantile(means, q)
    return float(interval[0]), float(interval[1])



def _seed_means(cases: Sequence[CaseResult], values: Sequence[float]) -> list[float]:
    if len(cases) != len(values):
        raise ValueError("case/value lengths do not match")
    grouped: dict[str, list[float]] = {}
    for row, value in zip(cases, values):
        grouped.setdefault(str(row.seed_id), []).append(float(value))
    return [
        sum(group) / float(len(group))
        for _, group in sorted(grouped.items())
    ]


@dataclass(frozen=True)
class Summary:
    case_count: int
    score_sum: float
    score_mean: float
    score_median: float
    score_p05: float
    score_min: float
    worst_decile_mean: float
    negative_cases: int
    negative_rate: float
    catastrophic_cases: int
    catastrophic_rate: float
    mse_ratio_mean: float
    bootstrap_mean_ci95: tuple[float, float]

    def to_dict(self) -> dict[str, object]:
        return {
            "case_count": self.case_count,
            "score_sum": self.score_sum,
            "score_mean": self.score_mean,
            "score_median": self.score_median,
            "score_p05": self.score_p05,
            "score_min": self.score_min,
            "worst_decile_mean": self.worst_decile_mean,
            "negative_cases": self.negative_cases,
            "negative_rate": self.negative_rate,
            "catastrophic_cases": self.catastrophic_cases,
            "catastrophic_rate": self.catastrophic_rate,
            "mse_ratio_mean": self.mse_ratio_mean,
            "bootstrap_mean_ci95": list(self.bootstrap_mean_ci95),
        }


@dataclass(frozen=True)
class Comparison:
    paired_cases: int
    mean_score_delta: float
    median_score_delta: float
    p05_score_delta: float
    min_score_delta: float
    win_tie_loss: tuple[int, int, int]
    bootstrap_delta_ci95: tuple[float, float]

    def to_dict(self) -> dict[str, object]:
        return {
            "paired_cases": self.paired_cases,
            "mean_score_delta": self.mean_score_delta,
            "median_score_delta": self.median_score_delta,
            "p05_score_delta": self.p05_score_delta,
            "min_score_delta": self.min_score_delta,
            "win_tie_loss": list(self.win_tie_loss),
            "bootstrap_delta_ci95": list(self.bootstrap_delta_ci95),
        }


@dataclass(frozen=True)
class Decision:
    promote: bool
    checks: Mapping[str, bool]

    def to_dict(self) -> dict[str, object]:
        return {"promote": self.promote, "checks": dict(self.checks)}


def summarize(cases: Sequence[CaseResult], rounds: int, seed: int) -> Summary:
    scores = [float(row.score) for row in cases]
    if not scores:
        raise ValueError("cannot summarize an empty case set")
    ordered = sorted(scores)
    tail = ordered[: max(1, (len(scores) + 9) // 10)]
    ratios = [float(row.mse_player) / max(float(row.mse_std), 1.0e-30) for row in cases]
    data = torch.tensor(scores, dtype=torch.float64)
    return Summary(
        len(scores),
        sum(scores),
        sum(scores) / len(scores),
        float(torch.median(data).item()),
        _percentile(scores, 0.05),
        min(scores),
        sum(tail) / len(tail),
        sum(score < 0.0 for score in scores),
        sum(score < 0.0 for score in scores) / len(scores),
        sum(score < -0.10 for score in scores),
        sum(score < -0.10 for score in scores) / len(scores),
        sum(ratios) / len(ratios),
        bootstrap_mean_ci(_seed_means(cases, scores), rounds, seed),
    )


def _key(row: CaseResult) -> tuple[object, ...]:
    return (row.seed_id, row.kind, row.scenario, row.test_index, row.causal, row.compute_dtype)


def compare(candidate: Sequence[CaseResult], champion: Sequence[CaseResult], rounds: int, seed: int) -> Comparison:
    left = {_key(row): row for row in candidate}
    right = {_key(row): row for row in champion}
    if set(left) != set(right) or len(left) != len(candidate) or len(right) != len(champion):
        raise ValueError("paired case keys do not match")
    ordered_keys = sorted(left, key=str)
    deltas = [left[key].score - right[key].score for key in ordered_keys]
    seed_deltas = _seed_means([left[key] for key in ordered_keys], deltas)
    wins = sum(delta > 1.0e-12 for delta in deltas)
    losses = sum(delta < -1.0e-12 for delta in deltas)
    return Comparison(
        len(deltas),
        sum(deltas) / len(deltas),
        float(torch.median(torch.tensor(deltas, dtype=torch.float64)).item()),
        _percentile(deltas, 0.05),
        min(deltas),
        (wins, len(deltas) - wins - losses, losses),
        bootstrap_mean_ci(seed_deltas, rounds, seed),
    )


def decide(
    candidate: Summary,
    incumbent: Summary | None,
    comparison: Comparison | None,
    candidate_timing: TimingResult,
    incumbent_timing: TimingResult | None,
    thresholds: Mapping[str, float],
    authoritative_timing: bool,
) -> Decision:
    checks: dict[str, bool] = {
        "mean_score": candidate.score_mean >= float(thresholds["min_mean_score"]),
        "negative_rate": candidate.negative_rate <= float(thresholds["max_negative_rate"]),
        "catastrophic_rate": candidate.catastrophic_rate <= float(thresholds["max_catastrophic_rate"]),
        "tail_score": candidate.worst_decile_mean >= float(thresholds["min_worst_decile_mean"]),
        "authoritative_timing": authoritative_timing,
    }
    if comparison is not None and incumbent is not None:
        checks["paired_delta"] = comparison.mean_score_delta >= float(thresholds["min_candidate_delta"])
        checks["paired_ci_lower"] = comparison.bootstrap_delta_ci95[0] >= float(thresholds["min_delta_ci_lower"])
        checks["negative_rate_delta"] = candidate.negative_rate - incumbent.negative_rate <= float(thresholds["max_negative_rate_delta"])
    if incumbent_timing is not None and incumbent_timing.player_quant_seconds > 0.0:
        checks["runtime"] = candidate_timing.player_quant_seconds / incumbent_timing.player_quant_seconds <= float(thresholds["max_runtime_ratio"])
    return Decision(all(checks.values()), checks)
