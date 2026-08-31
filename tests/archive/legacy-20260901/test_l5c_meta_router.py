from __future__ import annotations

import torch

import solution


def test_l5c_features_are_finite_and_operand_local_shaped() -> None:
    torch.manual_seed(921)
    weight = torch.randn(96, 128) * 0.05
    calibration = [torch.randn(31, 128) * 0.1, torch.randn(27, 128) * 0.1]
    gram = weight.t().mm(weight)
    features = solution._l5c_static_features(weight, calibration, gram)
    assert tuple(features.shape) == (8,)
    assert torch.isfinite(features).all()


def test_l5c_stump_finds_deterministic_split() -> None:
    features = torch.tensor([[0.0, 3.0], [1.0, 2.0], [4.0, 1.0], [5.0, 0.0]])
    labels = [0, 0, 2, 2]
    stump = solution._l5c_fit_stump(features, labels)
    assert stump[0] == 0
    assert solution._l5c_predict_stump(stump, features[0]) == 0
    assert solution._l5c_predict_stump(stump, features[-1]) == 2


def test_l5c_route_choice_is_bounded_and_crossfold_gated() -> None:
    torch.manual_seed(922)
    weight = torch.randn(192, 128) * 0.05
    calibration = [torch.randn(37, 128) * 0.1, torch.randn(29, 128) * 0.1]
    parent = solution._dense_to_hif4(weight)
    gram64 = solution._gram64(weight)
    deployment_gram = solution._dequantize_hif4(parent).t().mm(
        solution._dequantize_hif4(parent)
    )
    deployment_gram64 = solution._gram64(solution._dequantize_hif4(parent))
    route = solution._choose_l5c_route(
        weight,
        calibration,
        gram64,
        deployment_gram64,
        deployment_gram,
        None,
        True,
        False,
    )
    assert route in {-1, 0, 1, 2}


def test_l5c_state_route_is_scalar_and_dynamic_output_is_legal() -> None:
    torch.manual_seed(923)
    rows, channels, tokens = 192, 128, 23
    weight = torch.randn(rows, channels) * 0.05
    activation = [torch.randn(tokens, channels) * 0.1 for _ in range(2)]
    weight_pair = (weight.to(torch.bfloat16), torch.ones(rows, channels // 16))
    calibration_pairs = [
        (item.to(torch.bfloat16), torch.ones(tokens, channels // 16))
        for item in activation
    ]
    calibrated = solution.hif4_calibration_and_quantize_weight(
        *weight_pair, calibration_pairs
    )
    state = calibrated["activation_state"]
    assert isinstance(state["meta_route"], int)
    assert state["meta_route"] in {-1, 0, 1, 2}
    result = solution.hif4_dynamic_quantize_activation(
        activation[0].to(torch.bfloat16),
        torch.ones(tokens, channels // 16),
        state,
    )
    assert set(result) == {
        "scale_factor",
        "scale_lv2",
        "scale_lv3",
        "sign",
        "mant",
    }
    assert torch.isfinite(solution._dequantize_hif4(result)).all()
