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
    COMPACT_CASE_DESIGN,
    COMPACT_WINDOW_INDICES,
    DEFAULT_CASE_DESIGN,
    EFFECT_CASE_DESIGN,
    LINEAR_CASE_COUNT,
    NVFP4_INPUT_CODEC,
    PANEL_WINDOW_INDICES,
    PROTOCOL,
    REQUIRED_APIS,
    TEST_LENGTH,
    TEST_LENGTHS,
    AttentionCase,
    LinearCase,
    PreparedPack,
    Window,
    _attention,
    _attention_trace,
    _attention_decomposition_summary,
    _choose_cases,
    _depth_spread_indices,
    _linear_candidate_role_diagnostics,
    _linear_cross_holdout_consistency,
    _linear_decomposition_summary,
    _linear_error_source_details,
    _linear_generalization_summary,
    _linear_role_family,
    _paired_effect_diagnostics,
    _trend_diagnostics,
    _nvfp4_cache_profile,
    build_parser,
    load_nvfp4_cache,
    load_pack,
    load_solution,
    save_nvfp4_cache,
)
from reference_hif4 import encode_standard_hif4, decode_standard_hif4, validate_state  # noqa: E402
from nvfp4_sim import e4m3_round_up  # noqa: E402


def test_protocol_uses_stratified_real_wa_panel_by_default() -> None:
    assert PROTOCOL == "proxy-v2"
    assert NVFP4_INPUT_CODEC == "e4m3-subnormal-ceil-v1"
    assert CALIBRATION_LENGTHS == (10, 128, 512, 1024, 1024)
    assert TEST_LENGTH == 128
    assert TEST_LENGTHS == (10, 128, 512, 1024, 1024, 10, 128, 512, 1024, 1024, 128, 512)
    assert LINEAR_CASE_COUNT is None
    assert ATTENTION_CASE_COUNT is None
    assert DEFAULT_CASE_DESIGN == "stratified-real-wa-panel-v1"
    assert EFFECT_CASE_DESIGN == "paired-effect-panel-v1"
    assert COMPACT_CASE_DESIGN == "compact-generalization-panel-v1"
    assert COMPACT_WINDOW_INDICES == (1, 2, 6, 7)
    assert PANEL_WINDOW_INDICES == (0, 1, 2, 3, 4)


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
    assert len(linear_a) == pack.layers * 7
    assert len(attention_a) == pack.layers * len(PANEL_WINDOW_INDICES)
    assert len({(case.layer, case.role, case.test_window) for case in linear_a}) == len(linear_a)
    assert len({(case.layer, case.test_window) for case in attention_a}) == len(attention_a)
    assert {case.layer for case in linear_a} == set(range(pack.layers))
    assert {case.layer for case in attention_a} == set(range(pack.layers))
    assert {case.role for case in linear_a} == {"q", "k", "v", "o", "fc_gate", "fc_up", "proj"}
    assert {case.test_window for case in attention_a} == set(PANEL_WINDOW_INDICES)
    assert all(
        {case.test_window for case in linear_a if case.layer == layer} == set(PANEL_WINDOW_INDICES)
        for layer in range(pack.layers)
    )
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


def test_full_case_expansion_is_explicit_stress_only() -> None:
    pack = SimpleNamespace(
        layers=2,
        calibration_windows=[None] * len(CALIBRATION_LENGTHS),
        test_windows=[
            Window("validation", f"doc-{i}", 0, 0, 0, TEST_LENGTH, tuple(range(TEST_LENGTH)))
            for i in range(6)
        ],
    )
    linear, attention = _choose_cases(pack, full_cases=True)
    assert len(linear) == pack.layers * 7 * len(pack.test_windows)
    assert len(attention) == pack.layers * len(pack.test_windows)


def test_effect_panel_spans_depth_and_keeps_every_linear_role() -> None:
    pack = SimpleNamespace(
        layers=24,
        calibration_windows=[None] * len(CALIBRATION_LENGTHS),
        test_windows=[
            Window("validation" if i % 2 == 0 else "test", f"doc-{i}", 0, 0, 0, TEST_LENGTH, tuple(range(TEST_LENGTH)))
            for i in range(5)
        ],
        roles=("q", "k", "v", "o", "fc_gate", "fc_up", "proj"),
    )
    linear, attention = _choose_cases(pack, effect_panel=True)
    expected_layers = _depth_spread_indices(24, 8)
    assert expected_layers == (0, 3, 7, 10, 13, 16, 20, 23)
    assert len(linear) == 56
    assert {case.layer for case in linear} == set(expected_layers)
    assert all(
        {case.role for case in linear if case.layer == layer} == set(pack.roles)
        for layer in expected_layers
    )
    assert len(attention) == 5
    assert [case.test_window for case in attention] == [0, 1, 2, 3, 4]


def test_compact_panel_uses_cross_split_pairs_and_selected_depths() -> None:
    windows = [
        Window(
            "validation" if i % 2 == 0 else "test",
            f"doc-{i}",
            0,
            0,
            0,
            TEST_LENGTHS[i],
            tuple(range(TEST_LENGTHS[i])),
        )
        for i in range(len(TEST_LENGTHS))
    ]
    pack = SimpleNamespace(
        layers=24,
        calibration_windows=[None] * len(CALIBRATION_LENGTHS),
        test_windows=windows,
        roles=("q", "k", "v", "o", "fc_gate", "fc_up", "proj"),
    )
    linear, attention = _choose_cases(pack, compact_panel=True)
    expected_layers = _depth_spread_indices(24, 4)
    assert expected_layers == (0, 8, 15, 23)
    assert len(linear) == 4 * 7 * 2
    assert len(attention) == 4
    assert {case.layer for case in linear} == set(expected_layers)
    assert {case.test_window for case in linear} == set(COMPACT_WINDOW_INDICES)
    assert {tuple(case.calibration_indices) for case in linear} == {(1, 2)}
    for layer in expected_layers:
        for role in pack.roles:
            cases = [case for case in linear if case.layer == layer and case.role == role]
            assert len(cases) == 2
            assert {windows[case.test_window].split for case in cases} == {"validation", "test"}
            assert len({len(windows[case.test_window].input_ids) for case in cases}) == 1


def test_linear_generalization_summary_reports_tail_and_sources() -> None:
    items = [
        {
            "gain": gain,
            "mse_standard": 1.0,
            "mse_player": 1.0 - gain,
            "gain_w_only": gain / 2,
            "gain_a_only": gain / 3,
            "gain_both": gain,
            "interaction_gain": gain / 4,
            "layer": index,
            "role": "fc_gate",
            "test_split": "validation" if index % 2 == 0 else "test",
            "test_length": 128,
            "test_window": index,
        }
        for index, gain in enumerate((-0.2, 0.0, 0.1, 0.3))
    ]
    summary = _linear_generalization_summary(items)
    assert summary["gain"]["median"] == pytest.approx(0.05)
    assert summary["gain"]["worst_quartile_mean"] == pytest.approx(-0.2)
    assert summary["gain"]["negative_cases"] == 1
    assert summary["gain"]["positive_cases"] == 2
    assert summary["coverage"]["splits"] == ["test", "validation"]
    assert summary["component_gain"]["activation_only"]["count"] == 4


def test_cross_holdout_consistency_pairs_same_layer_role_and_length() -> None:
    items = [
        {"layer": 0, "role": "fc_gate", "test_length": 128, "test_split": "validation", "test_window": 6, "gain": 0.2},
        {"layer": 0, "role": "fc_gate", "test_length": 128, "test_split": "test", "test_window": 1, "gain": 0.1},
        {"layer": 8, "role": "proj", "test_length": 512, "test_split": "validation", "test_window": 2, "gain": 0.1},
        {"layer": 8, "role": "proj", "test_length": 512, "test_split": "test", "test_window": 7, "gain": -0.2},
    ]
    summary = _linear_cross_holdout_consistency(items)
    assert summary["enabled"] is True
    assert summary["pair_count"] == 2
    assert summary["same_sign_pairs"] == 1
    assert summary["opposite_or_zero_mismatch_pairs"] == 1
    assert summary["absolute_gap"]["maximum"] == pytest.approx(0.3)


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


def test_attention_trace_exposes_logits_and_probabilities() -> None:
    q = torch.randn(1, 5, 8 * 4)
    k = torch.randn(1, 5, 2 * 4)
    v = torch.randn(1, 5, 2 * 4)
    output, logits, probabilities = _attention_trace(q, k, v, 8, 2, 4)
    assert output.shape == (1, 5, 8 * 4)
    assert logits.shape == (1, 8, 5, 5)
    assert probabilities.shape == logits.shape
    assert torch.allclose(probabilities.sum(dim=-1), torch.ones_like(probabilities[..., 0]))


def test_linear_decomposition_reports_operand_and_interaction_sources() -> None:
    reference = torch.eye(2)
    standard = reference + 0.50
    weight_only = reference + 0.25
    activation_only = reference + 0.125
    both = reference + 0.10
    ref_weight = torch.eye(2)
    ref_activation = torch.eye(2)
    details = _linear_error_source_details(
        standard,
        weight_only,
        activation_only,
        both,
        reference,
        ref_weight,
        ref_weight,
        ref_weight,
        ref_activation,
        ref_activation,
        ref_activation,
    )
    assert details["gain_a_only"] > details["gain_w_only"]
    assert details["gain_both"] > details["gain_a_only"]
    assert details["interaction_gain"] < 0.0
    summary = _linear_decomposition_summary([details])
    assert summary["case_count"] == 1
    assert summary["gain"]["w_only"] == details["gain_w_only"]
    assert summary["interpretation"] == "activation_dominant"


def test_attention_decomposition_summary_contains_intermediate_sources() -> None:
    item = {
        "mse_standard": 2.0,
        "mse_q_only": 1.8,
        "mse_k_only": 1.7,
        "mse_v_only": 1.6,
        "mse_qk_only": 1.4,
        "mse_both": 1.2,
        "gain_q_only": 0.1,
        "gain_k_only": 0.15,
        "gain_v_only": 0.2,
        "gain_qk_only": 0.3,
        "gain_both": 0.4,
        "qk_interaction_gain": 0.05,
        "qkv_interaction_gain": 0.07,
        "logit_mse_standard": 0.03,
        "logit_mse_player": 0.02,
        "probability_mse_standard": 0.01,
        "probability_mse_player": 0.005,
        "probability_kl_standard_to_reference": 0.004,
        "probability_kl_player_to_reference": 0.002,
    }
    summary = _attention_decomposition_summary([item])
    assert summary["gain"]["qk_interaction"] == 0.05
    assert summary["intermediate"]["probability_mse_player"] == 0.005
    assert summary["interpretation"] == "qk_dominant"


def test_decomposition_is_enabled_by_default_and_can_be_disabled() -> None:
    assert build_parser().parse_args([]).decomposition is True
    assert build_parser().parse_args(["--no-decomposition"]).decomposition is False


def test_scenario_isolation_cli_is_mutually_exclusive() -> None:
    parser = build_parser()
    assert parser.parse_args([]).evaluation_scenario == "both"
    assert parser.parse_args(["--linear-only"]).evaluation_scenario == "linear"
    assert parser.parse_args(["--attention-only"]).evaluation_scenario == "attention"
    with pytest.raises(SystemExit):
        parser.parse_args(["--linear-only", "--attention-only"])


def test_paired_cli_options_are_explicit() -> None:
    args = build_parser().parse_args([
        "--candidate-json", "candidate.json",
        "--baseline-json", "baseline.json",
        "--focus-linear-roles", "fc_gate,fc_up",
        "--effect-panel",
    ])
    assert str(args.candidate_json).endswith("candidate.json")
    assert str(args.baseline_json).endswith("baseline.json")
    assert args.focus_linear_roles == "fc_gate,fc_up"
    assert args.effect_panel is True


def test_compact_panel_cli_is_explicit() -> None:
    assert build_parser().parse_args(["--compact-panel"]).compact_panel is True


def test_nvfp4_cache_cli_defaults_to_reuse() -> None:
    args = build_parser().parse_args([])
    assert args.nvfp4_cache is None
    assert args.nvfp4_cache_mode == "auto"
    assert build_parser().parse_args(["--nvfp4-cache-mode", "off"]).nvfp4_cache_mode == "off"


def test_nvfp4_prepared_cache_round_trip_and_profile_guard(tmp_path: Path) -> None:
    source = tmp_path / "dense.pt"
    source.write_bytes(b"dense-source-identity")
    window = Window("validation", "doc", 0, 0, 0, 2, (1, 2))
    pair = (torch.tensor([[0.0, 0.5]], dtype=torch.float32), torch.tensor([[1.0]], dtype=torch.float32))
    prepared = PreparedPack(
        weights=[{"q": pair}],
        linear_calibration_activations={"q": [[pair]]},
        test_activations={"q": [[pair]]},
        calibration_qkv=[[{}]],
        test_qkv=[[None]],
        calibration_windows=[window],
        linear_calibration_windows=[window],
        test_windows=[window],
        layers=1,
        hidden_size=2,
        q_heads=1,
        kv_heads=1,
        head_dim=2,
        linear_cases=[LinearCase(0, 0, "q", (0,), 0)],
        attention_cases=[AttentionCase(0, 0, (0,), 0)],
        metadata={"input_codec": NVFP4_INPUT_CODEC, "data_sha256": {"train": "abc"}},
        roles=("q",),
    )
    profile = _nvfp4_cache_profile(
        linear_count=None,
        attention_count=None,
        full_cases=False,
        effect_panel=False,
        compact_panel=True,
        evaluation_scenario="linear",
    )
    path = tmp_path / "prepared-nvfp4.pt"
    save_nvfp4_cache(prepared, path, source, profile)
    loaded = load_nvfp4_cache(path, source, profile)
    assert torch.equal(loaded.weights[0]["q"][0], pair[0])
    assert loaded.linear_cases == prepared.linear_cases
    assert loaded.metadata["nvfp4_cache_hit"] is True

    mismatched = dict(profile)
    mismatched["evaluation_scenario"] = "attention"
    with pytest.raises(RuntimeError, match="different evaluation profile"):
        load_nvfp4_cache(path, source, mismatched)


def test_static_linear_role_family_groups_qkv_fc_o_and_proj() -> None:
    assert [_linear_role_family(role) for role in ("q", "k", "v", "o", "fc_gate", "fc_up", "proj")] == [
        "qkv", "qkv", "qkv", "o", "fc", "fc", "proj"
    ]
    assert _linear_role_family("ffn_in") == "fc"


def test_candidate_role_diagnostics_pairs_same_case_against_v86() -> None:
    def result(name: str, gains: dict[str, float]) -> dict[str, object]:
        cases = []
        for role, gain in gains.items():
            cases.append({
                "layer": 0,
                "role": role,
                "role_family": _linear_role_family(role),
                "test_window": 0,
                "test_split": "validation",
                "test_length": 128,
                "gain": gain,
            })
        return {
            "candidate": name,
            "status": "ok",
            "decomposition": {"linear": {"enabled": True}},
            "case_scores": {"linear": cases},
        }

    diagnostics = _linear_candidate_role_diagnostics([
        result("v86", {"q": 0.1, "fc_gate": 0.2, "proj": 0.3}),
        result("candidate", {"q": 0.2, "fc_gate": 0.1, "proj": 0.0}),
    ])
    assert diagnostics["enabled"] is True
    assert diagnostics["baseline"] == "v86"
    candidate = diagnostics["candidates"]["candidate"]
    assert candidate["by_role"]["q"]["mean_delta_gain"] == 0.1
    assert candidate["by_role"]["fc_gate"]["mean_delta_gain"] == -0.1
    assert candidate["by_role_family"]["fc"]["mean_delta_gain"] == -0.1
    assert candidate["worst_cases"][0]["role"] == "proj"


def test_paired_effect_separates_focus_controls_and_error_sources() -> None:
    def linear(role: str, gain: float, player_mse: float, a_only: float) -> dict[str, object]:
        return {
            "layer": 0,
            "role": role,
            "role_family": _linear_role_family(role),
            "shape_bucket": "hidden_to_wide" if role.startswith("fc_") else "hidden_to_hidden",
            "test_window": 0,
            "test_split": "validation",
            "test_length": 128,
            "mse_standard": 1.0,
            "mse_player": player_mse,
            "reference_energy": 2.0,
            "relative_player_mse": player_mse / 2.0,
            "gain": gain,
            "gain_w_only": 0.1,
            "gain_a_only": a_only,
            "gain_both": gain,
            "interaction_gain": 0.0,
            "weight_relative_mse": 0.2,
            "activation_relative_mse": 0.3,
        }

    def attention(gain: float) -> dict[str, object]:
        return {
            "layer": 0,
            "test_window": 0,
            "test_split": "validation",
            "test_length": 10,
            "mse_standard": 1.0,
            "mse_player": 0.2,
            "reference_energy": 2.0,
            "relative_player_mse": 0.1,
            "gain": gain,
            "gain_q_only": 0.1,
            "gain_k_only": 0.1,
            "gain_v_only": 0.1,
            "gain_qk_only": 0.2,
            "gain_both": gain,
            "qk_interaction_gain": 0.0,
            "qkv_interaction_gain": 0.0,
            "logit_mse_player": 0.3,
            "probability_mse_player": 0.01,
            "probability_kl_player_to_reference": 0.02,
        }

    def result(name: str, fc_delta: float) -> dict[str, object]:
        linear_cases = [
            linear("q", 0.10, 0.90, 0.05),
            linear("fc_gate", 0.20 + fc_delta, 0.80 - fc_delta, 0.10 + fc_delta),
            linear("fc_up", 0.30 + fc_delta, 0.70 - fc_delta, 0.15 + fc_delta),
        ]
        return {
            "candidate": name,
            "status": "ok",
            "score": {"linear_mean": sum(item["gain"] for item in linear_cases) / 3, "attention_mean": 0.8, "overall_mean": 0.5},
            "timing": {
                "api_total_seconds": 10.0 + fc_delta,
                "api_seconds": {"hif4_dynamic_quantize_activation": 2.0 + fc_delta},
                "api_calls": {"hif4_dynamic_quantize_activation": 3},
            },
            "case_scores": {"linear": linear_cases, "attention": [attention(0.8)]},
        }

    diagnostics = _paired_effect_diagnostics(
        result("parent", 0.0), result("candidate", 0.05), ("fc",)
    )
    assert diagnostics["enabled"] is True
    assert diagnostics["linear"]["focus"]["effect"] == "consistent_improvement"
    assert diagnostics["linear"]["focus"]["positive_cases"] == 2
    assert diagnostics["linear"]["control"]["effect"] == "no_effect"
    assert diagnostics["linear"]["by_role_family"]["fc"]["component_delta_mean"]["a_only_gain"] == pytest.approx(0.05)
    assert diagnostics["attention"]["overall"]["effect"] == "no_effect"

    typo = _paired_effect_diagnostics(
        result("parent", 0.0), result("candidate", 0.05), ("unknown_role",)
    )
    assert typo["enabled"] is False
    assert "matched no Linear" in typo["reason"]


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
