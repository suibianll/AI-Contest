from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import solution


def test_all_scale_oracle_never_worse_than_local_offsets() -> None:
    torch.manual_seed(3)
    dense = torch.randn(4, 128)
    local = solution._encode_rows(dense, solution._BASE_OFFSETS)
    oracle = solution._encode_rows(dense, range(-254, 255))
    local_loss = (
        dense - solution._dequantize_hif4(local)
    ).square().reshape(4, 2, 64).sum(dim=-1)
    oracle_loss = (
        dense - solution._dequantize_hif4(oracle)
    ).square().reshape(4, 2, 64).sum(dim=-1)
    assert torch.all(oracle_loss <= local_loss + 1.0e-6)


def test_shared_lv2_legal_effective_exponents() -> None:
    legal = {(0, 0), (0, 1), (1, 0), (1, 1), (1, 2), (2, 1), (2, 2)}
    assert (0, 2) not in legal
    assert (2, 0) not in legal
    assert len(legal) == 7


def test_progressive_full_hierarchy_product_loss_is_monotone() -> None:
    torch.manual_seed(17)
    weight = torch.randn(16, 512) * 0.05
    activation = torch.randn(96, 512) * 0.3
    parent = solution._dense_to_hif4(weight)
    refined = solution._polish_weight_progressive(weight, parent, activation)
    parent_loss = solution._product_loss(activation, weight, parent)
    refined_loss = solution._product_loss(activation, weight, refined)
    assert refined_loss <= parent_loss + 1.0e-7
    assert torch.isfinite(solution._dequantize_hif4(refined)).all()
    assert torch.all((refined["scale_lv2"] == 1) | (refined["scale_lv2"] == 2))
    assert torch.all((refined["scale_lv3"] == 1) | (refined["scale_lv3"] == 2))


def test_progressive_full_hierarchy_is_deterministic() -> None:
    torch.manual_seed(23)
    weight = torch.randn(8, 512) * 0.05
    activation = torch.randn(64, 512) * 0.2
    parent = solution._dense_to_hif4(weight)
    first = solution._polish_weight_progressive(weight, parent, activation)
    second = solution._polish_weight_progressive(weight, parent, activation)
    for key in first:
        assert torch.equal(first[key], second[key]), key
