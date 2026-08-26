from __future__ import annotations

import pytest
import torch

from hif4_system.formats import (
    E6M2_LEVELS,
    dequantize_hif4,
    dequantize_nvfp4,
    standard_hif4_quantize,
    validate_hif4_params,
)


def test_e6m2_has_all_255_finite_official_values() -> None:
    assert E6M2_LEVELS.numel() == 255
    assert torch.unique(E6M2_LEVELS).numel() == 255
    assert E6M2_LEVELS[0].item() == 2.0 ** -48
    assert E6M2_LEVELS[-1].item() == 49152.0


def test_hif4_validator_rejects_non_e6m2_scale_and_fractional_mantissa() -> None:
    params = standard_hif4_quantize(torch.ones(1, 64))
    invalid_scale = {name: value.clone() for name, value in params.items()}
    invalid_scale["scale_factor"].fill_(1.1)
    with pytest.raises(ValueError, match="E6M2"):
        validate_hif4_params(invalid_scale, (1, 64))

    invalid_mantissa = {name: value.clone() for name, value in params.items()}
    invalid_mantissa["mant"].flatten()[0] = 0.1
    with pytest.raises(ValueError, match="multiple of 0.25"):
        validate_hif4_params(invalid_mantissa, (1, 64))


def test_standard_hierarchy_matches_exhaustive_group_search() -> None:
    torch.manual_seed(50)
    values = torch.randn(1, 64) * torch.exp(torch.randn(1, 64) * 1.5)
    params = standard_hif4_quantize(values)
    decoded = params["sign"] * params["mant"] * params["scale_lv3"] * params["scale_lv2"] * params["scale_factor"]
    grouped = values.reshape(1, 1, 8, 2, 4)
    scale = float(params["scale_factor"].item())
    for group_index in range(8):
        source = grouped[0, 0, group_index]
        actual_loss = float((decoded[0, 0, group_index] - source).square().sum())
        candidate_losses = []
        for lv2 in (1.0, 2.0):
            for lv3_left in (1.0, 2.0):
                for lv3_right in (1.0, 2.0):
                    lv3 = torch.tensor([[lv3_left], [lv3_right]])
                    local_scale = scale * lv2 * lv3
                    mantissa = torch.round(source.abs() * (4.0 / local_scale)).clamp(0, 7) * 0.25
                    reconstructed = source.sign() * mantissa * local_scale
                    candidate_losses.append(float((reconstructed - source).square().sum()))
        assert actual_loss == pytest.approx(min(candidate_losses), abs=1.0e-6)


def test_standard_hif4_shapes_match_logical_tensor() -> None:
    values = torch.randn(3, 128)

    params = standard_hif4_quantize(values)

    validate_hif4_params(params, values.shape)
    assert params["scale_factor"].shape == (3, 2, 1, 1, 1)
    assert params["scale_lv2"].shape == (3, 2, 8, 1, 1)
    assert params["scale_lv3"].shape == (3, 2, 8, 2, 1)
    assert params["sign"].shape == (3, 2, 8, 2, 4)
    assert dequantize_hif4(params).shape == values.shape


def test_standard_hif4_handles_extreme_and_zero_values() -> None:
    values = torch.tensor([[0.0] * 63 + [1.0e30]], dtype=torch.float32)

    params = standard_hif4_quantize(values)
    decoded = dequantize_hif4(params)

    validate_hif4_params(params, values.shape)
    assert torch.isfinite(decoded).all()
    assert torch.isfinite(params["scale_factor"]).all()


def test_nvfp4_dequantization_requires_block_aligned_last_dim() -> None:
    values = torch.ones(2, 16)
    scales = torch.full((2, 1), 0.5)

    assert torch.equal(dequantize_nvfp4(values, scales), torch.full((2, 16), 0.5, dtype=torch.bfloat16))

    with pytest.raises(ValueError, match="divisible"):
        dequantize_nvfp4(torch.ones(2, 15), torch.ones(2, 1))
