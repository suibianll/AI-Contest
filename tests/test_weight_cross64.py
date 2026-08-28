"""C40 robust block-LDLQ weight-refinement tests."""

from __future__ import annotations

import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import solution as sol  # noqa: E402


def _loss(
    quantized: torch.Tensor, dense: torch.Tensor, hessian: torch.Tensor
) -> torch.Tensor:
    error = quantized.to(torch.float32) - dense.to(torch.float32)
    return torch.einsum("ri,ij,rj->", error, hessian, error)


def _make_case(seed: int = 123) -> tuple[torch.Tensor, dict, torch.Tensor]:
    torch.manual_seed(seed)
    dense = torch.randn(32, 128) * 0.05
    params = sol._dense_to_hif4(
        dense,
        importance=None,
        group_gram=None,
        search_offsets=(-2, -1, 1, 2, 3),
        error_threshold=1.0e-7,
        accept_margin=0.005,
        max_refine_ratio=1.0,
    )
    hessian = torch.eye(128)
    cross = 0.85 * torch.eye(64)
    hessian[:64, 64:] = cross
    hessian[64:, :64] = cross
    return dense, params, hessian


def test_cross64_monotonic_complete_hessian_loss() -> None:
    dense, params, hessian = _make_case()
    old_flag = sol._WEIGHT_CROSS64
    try:
        sol._WEIGHT_CROSS64 = True
        parent = sol._dequantize_hif4(params).to(torch.float32)
        refined_params = sol._refine_weight_blocks_cross64(
            dense, params, hessian
        )
        refined = sol._dequantize_hif4(refined_params).to(torch.float32)
        assert _loss(refined, dense, hessian) <= _loss(
            parent, dense, hessian
        ) + 1.0e-6
        assert tuple(refined.shape) == tuple(dense.shape)
        assert torch.isfinite(refined).all()
    finally:
        sol._WEIGHT_CROSS64 = old_flag


def test_cross64_can_use_off_diagonal_coupling() -> None:
    dense, params, hessian = _make_case(seed=456)
    old_flag = sol._WEIGHT_CROSS64
    try:
        sol._WEIGHT_CROSS64 = True
        parent = sol._dequantize_hif4(params).to(torch.float32)
        refined_params = sol._refine_weight_blocks_cross64(
            dense, params, hessian
        )
        refined = sol._dequantize_hif4(refined_params).to(torch.float32)

        # A strongly correlated pair should expose a non-zero conditional
        # target correction.  The implementation may still find that the
        # parent is already optimal for a particular row, so assert a global
        # signal rather than requiring every row to change.
        assert torch.any((refined - parent).abs() > 0)
        assert _loss(refined, dense, hessian) < _loss(
            parent, dense, hessian
        )
    finally:
        sol._WEIGHT_CROSS64 = old_flag


def test_cross64_fold_objective_is_monotonic() -> None:
    dense, params, hessian = _make_case(seed=654)
    second_fold = torch.eye(128)
    second_fold[:64, 64:] = 0.65 * torch.eye(64)
    second_fold[64:, :64] = 0.65 * torch.eye(64)
    folds = torch.stack((hessian, second_fold), dim=0).unsqueeze(1)
    old_flag = sol._WEIGHT_CROSS64
    try:
        sol._WEIGHT_CROSS64 = True
        parent = sol._dequantize_hif4(params).to(torch.float32)
        refined_params = sol._refine_weight_blocks_cross64(
            dense,
            params,
            hessian,
            fold_pair_covariances=folds,
        )
        refined = sol._dequantize_hif4(refined_params).to(torch.float32)
        parent_losses = torch.stack(
            [_loss(parent, dense, fold) for fold in folds[:, 0]]
        )
        refined_losses = torch.stack(
            [_loss(refined, dense, fold) for fold in folds[:, 0]]
        )
        assert sol._cross64_robust_loss(
            refined_losses
        ) <= sol._cross64_robust_loss(parent_losses) + 1.0e-6
    finally:
        sol._WEIGHT_CROSS64 = old_flag


def test_cross64_fold_covariance_matches_direct_activation_gram() -> None:
    torch.manual_seed(246)
    samples = [torch.randn(24, 256), torch.randn(17, 256)]
    d = torch.linspace(0.75, 1.25, 256)
    permutation = torch.arange(255, -1, -1)
    result = sol._cross64_fold_pair_covariances(
        samples,
        d,
        permutation,
        block_smooth_size=64,
        block_smooth_seed=3,
    )
    assert result is not None
    assert tuple(result.shape) == (2, 2, 128, 128)
    for fold_index, sample in enumerate(samples):
        transformed = sol._linear_pair_transform(
            sample,
            d,
            permutation,
            64,
            3,
            weight_side=False,
        )
        grouped = transformed.reshape(int(sample.shape[0]), 2, 128)
        expected = torch.einsum(
            "rpi,rpj->pij", grouped, grouped
        ) / float(sample.shape[0])
        assert torch.allclose(result[fold_index], expected, atol=1.0e-5)


def test_cross64_disabled_is_exact_noop() -> None:
    dense, params, hessian = _make_case(seed=789)
    old_flag = sol._WEIGHT_CROSS64
    try:
        sol._WEIGHT_CROSS64 = False
        refined = sol._refine_weight_blocks_cross64(dense, params, hessian)
        for key in params:
            if torch.is_tensor(params[key]):
                assert torch.equal(refined[key], params[key])
            else:
                assert refined[key] == params[key]
    finally:
        sol._WEIGHT_CROSS64 = old_flag
