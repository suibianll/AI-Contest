from __future__ import annotations

import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "workbench"))

from joint_output_gauge_probe import (  # noqa: E402
    GAUGE_VALUES,
    _apply_group_gauge,
    _loo_folds,
    _paired_group_variants,
    _single_group_variants,
)


def test_gauge_values_are_the_fixed_quarter_power_set() -> None:
    assert len(GAUGE_VALUES) == 3
    assert GAUGE_VALUES[1] == 1.0
    assert abs(GAUGE_VALUES[0] * GAUGE_VALUES[2] - 1.0) < 1.0e-12


def test_single_group_variants_cover_each_block_group_and_option() -> None:
    values = torch.ones(2, 2, 64)
    variants, specs = _single_group_variants(values)
    assert tuple(variants.shape) == (2 * 8 * 3, 2, 2, 64)
    assert len(specs) == 48
    assert {item["block"] for item in specs} == {0, 1}
    assert {item["group"] for item in specs} == set(range(8))
    assert {item["option"] for item in specs} == {0, 1, 2}


def test_paired_group_gauge_preserves_a_w_product() -> None:
    torch.manual_seed(20260906)
    activation = torch.randn(5, 2, 64)
    weight = torch.randn(7, 2, 64)
    gauges = torch.tensor([
        [2.0 ** -0.25, 1.0, 2.0 ** 0.25, 1.0, 1.0, 1.0, 1.0, 1.0],
        [1.0, 1.0, 1.0, 2.0 ** 0.25, 1.0, 1.0, 1.0, 1.0],
    ])
    activation_g = _apply_group_gauge(activation, gauges, inverse=True)
    weight_g = _apply_group_gauge(weight, gauges, inverse=False)
    before = torch.einsum("nbd,rbd->nr", activation, weight)
    after = torch.einsum("nbd,rbd->nr", activation_g, weight_g)
    assert torch.allclose(before, after, rtol=1.0e-6, atol=1.0e-6)


def test_paired_variants_apply_inverse_gauge_to_k() -> None:
    q = torch.ones(1, 1, 64)
    k = torch.ones(1, 1, 64)
    q_variants, k_variants, specs = _paired_group_variants(q, k)
    assert tuple(q_variants.shape) == (24, 1, 1, 64)
    assert torch.allclose(q_variants[0, 0, 0, :8] * k_variants[0, 0, 0, :8], torch.ones(8))
    assert specs[0]["gauge"] < 1.0


def test_loo_folds_hold_out_each_window_once() -> None:
    folds = _loo_folds(5)
    assert len(folds) == 5
    assert [holdout for holdout, _ in folds] == list(range(5))
    assert all(len(train) == 4 and holdout not in train for holdout, train in folds)
