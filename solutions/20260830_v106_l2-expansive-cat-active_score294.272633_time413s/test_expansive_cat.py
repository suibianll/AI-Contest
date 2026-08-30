from __future__ import annotations

import torch

import solution


def test_expansive_cat_is_structurally_routed_and_product_preserving() -> None:
    torch.manual_seed(3)
    weight = torch.randn(4864, 896) * 0.05
    calibration = [torch.randn(64, 896) * 0.1, torch.randn(64, 896) * 0.1]
    base, seed, block_size = solution._choose_boat(weight, calibration)
    candidate = solution._choose_expansive_cat_balance(
        weight, calibration, base, seed, block_size
    )

    assert torch.isfinite(candidate).all()
    assert candidate.shape == base.shape
    for sample in calibration[:1]:
        left = solution._apply_boat_rotation(
            sample / candidate, seed, block_size
        )
        right = solution._apply_boat_rotation(
            weight[: sample.shape[0]] * candidate, seed, block_size
        )
        # The balanced frame preserves the unquantized product; this uses a
        # small row slice only to keep the synthetic assertion inexpensive.
        reference = sample @ weight[: sample.shape[0]].t()
        transformed = left @ right.t()
        assert torch.allclose(transformed, reference, rtol=2.0e-4, atol=2.0e-4)


def test_expansive_cat_skips_non_expansive_shapes() -> None:
    torch.manual_seed(4)
    weight = torch.randn(896, 4864)
    calibration = [torch.randn(32, 4864)]
    base, seed, block_size = solution._choose_boat(weight, calibration)
    candidate = solution._choose_expansive_cat_balance(
        weight, calibration, base, seed, block_size
    )
    assert torch.equal(candidate, base)
