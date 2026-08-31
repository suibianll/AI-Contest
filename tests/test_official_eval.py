from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "evaluator"))

from official_eval import (  # noqa: E402
    ATTENTION_CASE_COUNT,
    CALIBRATION_LENGTHS,
    LINEAR_CASE_COUNT,
    PROTOCOL,
    REQUIRED_APIS,
    TEST_LENGTH,
    Window,
    _attention,
    _choose_cases,
    load_solution,
)
from reference_hif4 import encode_standard_hif4, decode_standard_hif4, validate_state  # noqa: E402


def test_protocol_has_one_official_case_shape() -> None:
    assert PROTOCOL == "official-shape-v1"
    assert CALIBRATION_LENGTHS == (10, 128, 512, 1024, 1024)
    assert TEST_LENGTH == 128
    assert (LINEAR_CASE_COUNT, ATTENTION_CASE_COUNT) == (250, 200)


def test_case_selection_is_deterministic_and_has_no_duplicate_tuples() -> None:
    pack = SimpleNamespace(
        layers=24,
        test_windows=[
            Window("validation", f"doc-{i}", 0, 0, 0, TEST_LENGTH, tuple(range(TEST_LENGTH)))
            for i in range(9)
        ],
    )
    linear_a, attention_a = _choose_cases(pack)
    linear_b, attention_b = _choose_cases(pack)
    assert linear_a == linear_b
    assert attention_a == attention_b
    assert len(linear_a) == LINEAR_CASE_COUNT
    assert len(attention_a) == ATTENTION_CASE_COUNT
    assert len(set(linear_a)) == LINEAR_CASE_COUNT
    assert len(set(attention_a)) == ATTENTION_CASE_COUNT


@pytest.mark.parametrize("tokens", [10, 128, 512, 1024])
def test_attention_kernel_accepts_each_official_sequence_length(tokens: int) -> None:
    q = torch.randn(tokens, 8 * 4)
    k = torch.randn(tokens, 2 * 4)
    v = torch.randn(tokens, 2 * 4)
    result = _attention(q[None], k[None], v[None], q_heads=8, kv_heads=2, head_dim=4)
    assert result.shape == (1, tokens, 8 * 4)
    assert torch.isfinite(result).all()


def test_solution_loader_finds_the_six_public_apis() -> None:
    module = load_solution(ROOT / "solution.py")
    assert all(callable(getattr(module, name)) for name in REQUIRED_APIS)


def test_reference_state_rules_remain_independent_of_candidate() -> None:
    validate_state({"lengths": list(CALIBRATION_LENGTHS), "tensor": torch.ones(2, dtype=torch.float32)})
    with pytest.raises(ValueError, match="finite"):
        validate_state(float("nan"))


def test_reference_codec_round_trip_has_expected_logical_shape() -> None:
    dense = torch.randn(3, 64)
    params = encode_standard_hif4(dense)
    restored = decode_standard_hif4(params)
    assert restored.shape == dense.shape
    assert torch.isfinite(restored).all()
