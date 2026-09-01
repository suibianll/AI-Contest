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
    NVFP4_INPUT_CODEC,
    PROTOCOL,
    REQUIRED_APIS,
    TEST_LENGTH,
    TEST_LENGTHS,
    Window,
    _attention,
    _choose_cases,
    _trend_diagnostics,
    load_pack,
    load_solution,
)
from reference_hif4 import encode_standard_hif4, decode_standard_hif4, validate_state  # noqa: E402
from nvfp4_sim import e4m3_round_up  # noqa: E402


def test_protocol_uses_full_real_wa_by_default() -> None:
    assert PROTOCOL == "proxy-v2"
    assert NVFP4_INPUT_CODEC == "e4m3-subnormal-ceil-v1"
    assert CALIBRATION_LENGTHS == (10, 128, 512, 1024, 1024)
    assert TEST_LENGTH == 128
    assert TEST_LENGTHS == (10, 128, 512, 1024, 1024, 10, 128, 512, 1024, 1024, 128, 512)
    assert LINEAR_CASE_COUNT is None
    assert ATTENTION_CASE_COUNT is None


def test_case_selection_is_deterministic_and_has_no_duplicate_tuples() -> None:
    pack = SimpleNamespace(
        layers=24,
        calibration_windows=[None] * len(CALIBRATION_LENGTHS),
        test_windows=[
            Window("validation", f"doc-{i}", 0, 0, 0, TEST_LENGTH, tuple(range(TEST_LENGTH)))
            for i in range(12)
        ],
    )
    linear_a, attention_a = _choose_cases(pack)
    linear_b, attention_b = _choose_cases(pack)
    assert linear_a == linear_b
    assert attention_a == attention_b
    assert LINEAR_CASE_COUNT is None
    assert ATTENTION_CASE_COUNT is None
    assert len(linear_a) == pack.layers * 7 * len(pack.test_windows)
    assert len(attention_a) == pack.layers * len(pack.test_windows)
    assert len({(case.layer, case.role, case.test_window) for case in linear_a}) == len(linear_a)
    assert len({(case.layer, case.test_window) for case in attention_a}) == len(attention_a)
    assert {case.layer for case in linear_a} == set(range(pack.layers))
    assert {case.layer for case in attention_a} == set(range(pack.layers))
    assert {case.role for case in linear_a} == {"q", "k", "v", "o", "fc_gate", "fc_up", "proj"}
    assert all(len(case.calibration_indices) == 2 for case in linear_a)
    assert {tuple(case.calibration_indices) for case in linear_a} == {(0, 1)}
    assert all(tuple(case.calibration_indices) == tuple(range(5)) for case in attention_a)


def test_case_limits_are_explicit_smoke_only() -> None:
    pack = SimpleNamespace(
        layers=2,
        calibration_windows=[None] * len(CALIBRATION_LENGTHS),
        test_windows=[
            Window("validation", f"doc-{i}", 0, 0, 0, TEST_LENGTH, tuple(range(TEST_LENGTH)))
            for i in range(3)
        ],
    )
    linear, attention = _choose_cases(pack, linear_count=5, attention_count=4)
    assert len(linear) == 5
    assert len(attention) == 4


def test_trend_diagnostics_does_not_fit_or_rewrite_scores() -> None:
    results = [
        {"candidate": "v86", "status": "ok", "official": {"score": 16744, "cohort": "new-weight"}, "score": {"overall_mean": 0.4}},
        {"candidate": "v147", "status": "ok", "official": {"score": 16579, "cohort": "new-weight"}, "score": {"overall_mean": 0.5}},
    ]
    diagnostics = _trend_diagnostics(results)
    assert diagnostics["status"] == "inversion_detected"
    assert diagnostics["inverted_pairs"] == 1
    assert results[0]["score"]["overall_mean"] == 0.4


def test_nvfp4_scale_preserves_e4m3_subnormal_range() -> None:
    values = torch.tensor([2.0 ** -10, 2.0 ** -9, 2.0 ** -8, 2.0 ** -6], dtype=torch.float32)
    rounded = e4m3_round_up(values)
    assert torch.equal(rounded, torch.tensor([2.0 ** -9, 2.0 ** -9, 2.0 ** -8, 2.0 ** -6]))


def test_legacy_cache_is_not_accepted_by_proxy_v2() -> None:
    path = ROOT / "artifacts" / "official_eval" / "cache" / "qwen2.5-0.5b-official-shape-v1.pt"
    with pytest.raises(RuntimeError, match="diagnostic-only"):
        load_pack(path)


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
