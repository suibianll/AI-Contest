from __future__ import annotations

import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "workbench"))

import legal_codec_output_probe as probe  # noqa: E402


def test_e4m3_enumeration_is_complete() -> None:
    values = probe.finite_e4m3_scales()
    assert len(values) == 126
    assert values[0] == 2.0 ** -9
    assert values[-1] == 448.0


def test_signed_carriers_are_counted_by_absolute_code() -> None:
    counts = probe.carrier_code_counts(torch.tensor([-1.0, 1.0, -0.5, 0.5, 0.0]))
    assert counts["1.0"] == 2
    assert counts["0.5"] == 2
    assert counts["zero"] == 1


def test_compatibility_uses_input_scale_and_preserves_sign() -> None:
    exact = probe.compatibility_exact_mask(
        torch.tensor([-1.0]), torch.tensor([0.25]),
        torch.tensor([0.25]), torch.tensor([1.0]),
        torch.tensor([1.0]), torch.tensor([1.0]),
    )
    missing_scale = probe.compatibility_exact_mask(
        torch.tensor([-1.0]), torch.tensor([1.0]),
        torch.tensor([0.25]), torch.tensor([1.0]),
        torch.tensor([1.0]), torch.tensor([1.0]),
    )
    assert bool(exact.item())
    assert not bool(missing_scale.item())


def test_exact_solver_returns_official_shared_hierarchy() -> None:
    torch.manual_seed(7)
    params, loss = probe.exact_legal_blocks(torch.randn(3, 64))
    assert loss.shape == (3,)
    assert tuple(params["scale_lv2"].shape) == (3, 1, 8, 1, 1)
    assert tuple(params["scale_lv3"].shape) == (3, 1, 8, 2, 1)
    assert tuple(params["mant"].shape) == (3, 1, 8, 2, 4)


def test_operand_and_output_targets_are_distinct() -> None:
    checks = probe._regression_checks()
    assert checks["operand_vs_output_objective"]["pass"]


def test_r0_regressions_all_pass() -> None:
    assert probe._regression_checks()["all_pass"]
