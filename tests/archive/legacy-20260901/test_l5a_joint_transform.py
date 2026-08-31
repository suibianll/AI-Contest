from __future__ import annotations

import torch

import solution


def test_l5a_block_permutations_are_valid_and_block_local() -> None:
    pressure = torch.linspace(0.0, 3.0, 128)
    identity = solution._l5a_block_permutation(pressure, 0)
    sorted_order = solution._l5a_block_permutation(pressure, 1)
    zigzag = solution._l5a_block_permutation(pressure, 2)
    quartile = solution._l5a_block_permutation(pressure, 3)
    expected = torch.arange(128)
    assert torch.equal(identity, expected)
    for order in (sorted_order, zigzag, quartile):
        assert torch.equal(torch.sort(order).values, expected)
        # Every output position stays within its original 64-channel block.
        assert torch.equal(order // 64, expected // 64)


def test_l5a_equivalent_transform_preserves_unquantized_product() -> None:
    torch.manual_seed(810)
    rows, channels = 37, 128
    weight = torch.randn(rows, channels) * 0.05
    activation = torch.randn(19, channels) * 0.1
    balance = torch.linspace(0.75, 1.25, channels)
    pressure = torch.linspace(1.0, 0.0, channels)
    permutation = solution._l5a_block_permutation(pressure, 2)
    transformed_activation = solution._apply_boat_rotation(
        activation / balance,
        seed=1,
        block_size=64,
        permutation=permutation,
    )
    transformed_weight = solution._apply_boat_rotation(
        weight * balance,
        seed=1,
        block_size=64,
        permutation=permutation,
    )
    reference = activation @ weight.t()
    transformed = transformed_activation @ transformed_weight.t()
    assert torch.allclose(transformed, reference, rtol=2.0e-4, atol=2.0e-4)


def test_l5a_selector_is_cross_fold_gated() -> None:
    torch.manual_seed(811)
    weight = torch.randn(192, 128) * 0.05
    calibration = [torch.randn(48, 128) * 0.1, torch.randn(41, 128) * 0.1]
    balance = torch.ones(128)
    selected = solution._choose_l5a_permutation(
        weight, calibration, balance, seed=1, block_size=64
    )
    if selected is not None:
        identity = torch.arange(128)
        assert torch.equal(torch.sort(selected).values, identity)
        assert torch.equal(selected // 64, identity // 64)


def test_dynamic_transform_accepts_l5a_state_permutation() -> None:
    torch.manual_seed(812)
    channels = 128
    dense = torch.randn(13, channels) * 0.1
    state = {
        "smooth_inv": torch.linspace(0.8, 1.2, channels),
        "permutation": solution._l5a_block_permutation(
            torch.linspace(0.0, 1.0, channels), 1
        ).to(torch.int32),
        "block_smooth_size": 64,
        "block_smooth_seed": 0,
        "gram64": None,
        "deployment_gram64": None,
        "final_gram_route": False,
        "gals_final": False,
        "deployment_gram": None,
        "global_lrh": None,
    }
    # Feed a legal NVFP4-shaped pair through the public dynamic entry point.
    quant = dense.to(torch.bfloat16)
    scale = torch.ones(13, channels // 16)
    result = solution.hif4_dynamic_quantize_activation(quant, scale, state)
    assert set(result) == {
        "scale_factor",
        "scale_lv2",
        "scale_lv3",
        "sign",
        "mant",
    }
