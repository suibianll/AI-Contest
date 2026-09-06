from __future__ import annotations

import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "workbench"))

from hierarchy_partition_activation_probe import (  # noqa: E402
    _pressure_order,
    _restore_permuted_blocks,
)


def test_pressure_order_is_a_bijection() -> None:
    activation = torch.arange(128, dtype=torch.float32).reshape(2, 64) + 1.0
    weight = torch.flip(activation, dims=[1])
    order, activation_rms, weight_rms = _pressure_order(activation, weight)
    assert sorted(order.tolist()) == list(range(64))
    assert torch.isfinite(activation_rms).all()
    assert torch.isfinite(weight_rms).all()


def test_restore_permuted_blocks_recovers_original_coordinate_order() -> None:
    order_a = torch.roll(torch.arange(64, dtype=torch.int64), 7)
    order_b = torch.flip(torch.arange(64, dtype=torch.int64), dims=[0])
    original = torch.arange(512, dtype=torch.float32).reshape(4, 2, 64)
    permuted = torch.stack(
        [original[:, 0].index_select(-1, order_a), original[:, 1].index_select(-1, order_b)],
        dim=0,
    )
    restored = _restore_permuted_blocks(permuted, [order_a, order_b])
    assert torch.equal(restored, original.permute(1, 0, 2))


def test_restore_rejects_non_block_width() -> None:
    bad = torch.zeros(1, 2, 8)
    try:
        _restore_permuted_blocks(bad, [torch.arange(8)])
    except ValueError:
        pass
    else:
        raise AssertionError("expected a 64-channel block width check")
