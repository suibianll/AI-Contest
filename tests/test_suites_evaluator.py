from __future__ import annotations

from pathlib import Path

import pytest
import torch

from hif4_system.config import load_config
from hif4_system.evaluator import evaluate_solution
from hif4_system.solution_loader import load_solution
from hif4_system.suites import build_suite


ROOT = Path(__file__).resolve().parents[1]


def test_suite_is_deterministic_on_cpu() -> None:
    config = load_config(None)
    tier = config.tier("smoke")
    left = build_suite(101, tier, torch.device("cpu"), tier_name="smoke", split="dev")
    right = build_suite(101, tier, torch.device("cpu"), tier_name="smoke", split="dev")

    assert torch.equal(left.linear[0].weight[0], right.linear[0].weight[0])
    assert torch.equal(left.attention[0].calibration[0]["q"][0], right.attention[0].calibration[0]["q"][0])


def test_minimal_solution_produces_linear_and_attention_cases() -> None:
    config = load_config(None)
    suite = build_suite(101, config.tier("smoke"), torch.device("cpu"), tier_name="smoke", split="dev")
    api = load_solution(ROOT / "tests" / "fixtures" / "minimal_solution.py")

    result = evaluate_solution(
        api,
        suite,
        torch.device("cpu"),
        compute_dtypes=("fp32",),
        causal_modes=(False, True),
    )

    assert {row.kind for row in result.cases} == {"linear", "attention"}
    assert len(result.cases) > 0
    assert result.timing.wall_seconds >= result.timing.player_quant_seconds


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available in this environment")
def test_solution_attention_calibration_runs_on_cuda() -> None:
    config = load_config(None)
    suite = build_suite(101, config.tier("smoke"), torch.device("cuda"), tier_name="smoke", split="dev")
    api = load_solution(ROOT / "solution.py")

    result = evaluate_solution(
        api,
        suite,
        torch.device("cuda"),
        compute_dtypes=("fp32",),
        causal_modes=(False, True),
    )

    assert result.cases



def test_attention_shapes_are_not_bound_to_scenarios() -> None:
    config = load_config(None)
    tier = config.tier("standard")
    first = build_suite(101, tier, torch.device("cpu"), tier_name="standard", split="dev")
    second = build_suite(211, tier, torch.device("cpu"), tier_name="standard", split="dev")

    assert [case.name for case in first.attention] != [case.name for case in second.attention]
    assert {case.head_dim for case in first.attention} == {64, 128}


def test_holdout_uses_an_independent_distribution_stream() -> None:
    config = load_config(None)
    tier = config.tier("standard")
    dev = build_suite(101, tier, torch.device("cpu"), tier_name="standard", split="dev")
    holdout = build_suite(101, tier, torch.device("cpu"), tier_name="standard", split="holdout")

    assert not torch.equal(dev.linear[0].weight[0], holdout.linear[0].weight[0])
