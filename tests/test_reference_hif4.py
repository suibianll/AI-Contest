from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "evaluator"))

from reference_hif4 import (  # noqa: E402
    decode_standard_hif4,
    encode_standard_hif4,
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


def assert_bitwise_equal_to_solution(dense: torch.Tensor) -> None:
    solution = load_solution()
    expected = solution_standard(solution, dense)
    actual = decode_standard_hif4(encode_standard_hif4(dense)).to(torch.float32)
    assert actual.shape == expected.shape
    assert torch.equal(actual, expected)


def test_reference_codec_matches_solution_standard_path_bitwise() -> None:
    torch.manual_seed(101)
    assert_bitwise_equal_to_solution(torch.randn(4, 128) * 0.05)
    assert_bitwise_equal_to_solution(torch.randn(3, 7, 256) * 300.0)
    assert_bitwise_equal_to_solution(torch.randn(64) * 1.0e-8)
    assert_bitwise_equal_to_solution(torch.randn(2, 64).abs() * 49152.0)


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
    # Bitwise parity with the solution standard path (which applies the same
    # nan_to_num clamping before quantization).
    assert_bitwise_equal_to_solution(dense)


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
