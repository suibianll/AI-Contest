from __future__ import annotations

import torch

import solution


def test_schur_pairs_are_ranked_and_disjoint_first() -> None:
    torch.manual_seed(901)
    matrix = torch.randn(256, 256)
    matrix = matrix.t().mm(matrix)
    pairs = solution._select_schur_pairs(matrix)
    assert 0 < len(pairs) <= solution._L5B_SCHUR_MAX_PAIRS
    assert len({block for pair in pairs for block in pair}) == 2 * len(pairs)
    for left, right in pairs:
        assert 0 <= left < right < 4


def test_schur_gram_is_symmetric_psd_and_block_shaped() -> None:
    torch.manual_seed(902)
    matrix = torch.randn(256, 256)
    matrix = matrix.t().mm(matrix)
    pairs = [(0, 1), (2, 3)]
    result = solution._build_schur_gram(matrix, pairs)
    assert result is not None
    assert tuple(result.shape) == (4, 64, 64)
    assert torch.allclose(result, result.transpose(-1, -2), atol=1.0e-5)
    minimum = torch.linalg.eigvalsh(result).amin()
    assert float(minimum) >= -1.0e-5


def test_sparse_weight_candidate_changes_only_selected_blocks() -> None:
    torch.manual_seed(903)
    weight = torch.randn(128, 256) * 0.05
    activation = torch.randn(97, 256) * 0.1
    parent = solution._dense_to_hif4(weight)
    candidate = solution._l5b_sparse_weight_candidate(weight, parent, activation)
    parent_q = solution._dequantize_hif4(parent)
    candidate_q = solution._dequantize_hif4(candidate)
    changed = (candidate_q - parent_q).abs().sum(dim=0) > 1.0e-9
    # The helper is allowed to reject the proposal, but if it emits a change
    # it must touch no more than the two selected 64-channel pairs.
    assert int(changed.sum()) <= 4 * solution._BLOCK


def test_activation_schur_gate_is_two_fold_and_finite() -> None:
    torch.manual_seed(904)
    weight = torch.randn(128, 256) * 0.05
    deployment_gram = weight.t().mm(weight)
    calibration = [torch.randn(64, 256) * 0.1, torch.randn(53, 256) * 0.1]
    pairs, schur, pair_hessian = solution._choose_l5b_activation_schur(
        deployment_gram, solution._gram64(weight), calibration
    )
    if pairs is not None:
        assert tuple(pairs.shape) == (solution._L5B_SCHUR_MAX_PAIRS, 2)
        assert schur is not None
        assert pair_hessian is not None
        assert tuple(pair_hessian.shape) == (
            solution._L5B_SCHUR_MAX_PAIRS,
            2 * solution._BLOCK,
            2 * solution._BLOCK,
        )
        assert torch.isfinite(schur).all()
        assert torch.isfinite(pair_hessian).all()
        flat = pairs.to(torch.int64).reshape(-1)
        assert int(torch.unique(flat).numel()) == int(flat.numel())


def test_hif4_calibration_state_includes_sparse_fields() -> None:
    torch.manual_seed(905)
    channels = 256
    weight = torch.randn(128, channels) * 0.05
    activation = torch.randn(32, channels) * 0.1
    # The public calibration API expects NVFP4-shaped tensors.  Unit values
    # are sufficient here because this test checks state shape/validation.
    result = solution.hif4_calibration_and_quantize_weight(
        weight.to(torch.bfloat16),
        torch.ones(128, channels // 16),
        [(activation.to(torch.bfloat16), torch.ones(32, channels // 16))] * 2,
    )
    state = result["activation_state"]
    assert (
        "schur_pairs" in state
        and "schur_gram64" in state
        and "schur_pair_hessian" in state
    )
    if state["schur_pairs"] is not None:
        assert state["schur_pairs"].device.type == "cpu"
        assert state["schur_pairs"].dtype == torch.int32
        assert state["schur_pair_hessian"] is not None
        assert state["schur_pair_hessian"].device.type == "cpu"
