from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "evaluator"))

from reference_hif4 import (  # noqa: E402
    dequantize_hif4,
    dequantize_nvfp4,
    decode_standard_hif4,
    encode_standard_hif4,
    validate_hif4_params,
    validate_state,
)


def load_solution() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "reference_hif4_solution", ROOT / "solution.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def solution_standard(solution: ModuleType, dense: torch.Tensor) -> torch.Tensor:
    params = solution._dense_to_hif4(dense, search_offsets=())
    return solution._dequantize_hif4(params).to(torch.float32)


def assert_no_worse_than_threshold_solution(dense: torch.Tensor) -> None:
    solution = load_solution()
    expected = solution_standard(solution, dense)
    actual = decode_standard_hif4(encode_standard_hif4(dense)).to(torch.float32)
    assert actual.shape == expected.shape
    actual_mse = (actual - dense.to(torch.float32)).square().mean()
    threshold_mse = (expected - dense.to(torch.float32)).square().mean()
    assert actual_mse <= threshold_mse + 1.0e-12


def test_reference_codec_uses_mse_optimal_hierarchy() -> None:
    torch.manual_seed(101)
    assert_no_worse_than_threshold_solution(torch.randn(4, 128) * 0.05)
    assert_no_worse_than_threshold_solution(torch.randn(3, 7, 256) * 300.0)
    assert_no_worse_than_threshold_solution(torch.randn(64) * 1.0e-8)
    assert_no_worse_than_threshold_solution(torch.randn(2, 64).abs() * 49152.0)


def test_reference_codec_handles_nonfinite_inputs() -> None:
    dense = torch.tensor(
        [
            float("nan"),
            float("inf"),
            float("-inf"),
            0.0,
            -0.0,
            1.0,
        ]
        * 11,  # 66 values, last two padded with finite values
    )
    dense = torch.cat((dense[:64], torch.randn(64) * 0.1))
    params = encode_standard_hif4(dense)
    decoded = decode_standard_hif4(params)
    assert torch.isfinite(decoded).all()
    assert torch.isfinite(decoded).all()


def test_reference_codec_is_deterministic() -> None:
    torch.manual_seed(103)
    dense = torch.randn(5, 192) * 0.7
    first = decode_standard_hif4(encode_standard_hif4(dense))
    second = decode_standard_hif4(encode_standard_hif4(dense.clone()))
    assert torch.equal(first, second)


def test_reference_codec_rejects_invalid_shapes() -> None:
    try:
        encode_standard_hif4(torch.randn(48))
    except ValueError as error:
        assert "divisible" in str(error)
    else:  # pragma: no cover
        raise AssertionError("expected ValueError for non-multiple-of-64 input")


def test_reference_codec_params_are_canonical() -> None:
    torch.manual_seed(105)
    dense = torch.randn(4, 64)
    params = encode_standard_hif4(dense)
    # Canonical zero: sign is exactly zero wherever the mantissa is zero.
    zero_mant = params["mant"] == 0.0
    assert (params["sign"][zero_mant] == 0.0).all()
    # Frozen hierarchy shapes.
    assert tuple(params["scale_factor"].shape) == (4, 1, 1, 1, 1)
    assert tuple(params["scale_lv2"].shape) == (4, 1, 8, 1, 1)
    assert tuple(params["scale_lv3"].shape) == (4, 1, 8, 2, 1)
    assert params["scale_lv2"].dtype == torch.float32
    # Standard scale codes are finite E6M2 values (no offset search here).
    assert (params["scale_factor"] > 0).all()


def test_official_nvfp4_dequantization_has_bf16_rounding() -> None:
    values = torch.tensor([[1.1] * 16 + [-0.7] * 16], dtype=torch.float32)
    scales = torch.tensor([[0.3, 2.0]], dtype=torch.float32)
    actual = dequantize_nvfp4(values, scales)
    expected = (values.unflatten(-1, (-1, 16)) * scales.unsqueeze(-1)).flatten(-2)
    assert actual.dtype == torch.bfloat16
    assert torch.equal(actual, expected.to(torch.bfloat16))


def test_candidate_hif4_is_independently_validated_and_dequantized() -> None:
    torch.manual_seed(107)
    dense = torch.randn(2, 128)
    params = encode_standard_hif4(dense)
    validate_hif4_params(params, dense.shape)
    actual = dequantize_hif4(params, dense.shape)
    assert torch.equal(actual, decode_standard_hif4(params).to(torch.float32))

    invalid = dict(params)
    invalid["mant"] = params["mant"].clone()
    invalid["mant"].reshape(-1)[0] = 0.3
    try:
        validate_hif4_params(invalid, dense.shape)
    except ValueError as error:
        assert "multiple of 0.25" in str(error)
    else:  # pragma: no cover
        raise AssertionError("expected invalid HiF4 mantissa to be rejected")


def test_state_validation_enforces_official_tree_contract() -> None:
    validate_state({"scale": 1.0, "tensor": torch.ones(4, dtype=torch.float32)})
    try:
        validate_state({"bad": torch.ones(1, dtype=torch.float64)})
    except ValueError as error:
        assert "dtype" in str(error)
    else:  # pragma: no cover
        raise AssertionError("expected float64 state tensor to be rejected")
