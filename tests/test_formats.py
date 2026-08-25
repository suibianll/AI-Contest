from __future__ import annotations

import pytest
import torch

from hif4_system.formats import (
    dequantize_hif4,
    dequantize_nvfp4,
    standard_hif4_quantize,
    validate_hif4_params,
)


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
