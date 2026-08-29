"""Focused tests for the fixed-Q(A) JDRQ weight-only search."""

from __future__ import annotations

import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "evaluator"))

import solution  # noqa: E402
from nvfp4_sim import nvfp4_encode  # noqa: E402


def test_jdrq_dual_projection_matches_primal_ridge() -> None:
    torch.manual_seed(7)
    z = torch.randn(19, 11)
    residual = torch.randn(19, 5)
    lam = 0.17
    projection = solution._jdrq_ridge_projection(z, residual, lam)
    scale = z.square().mean()
    regularization = lam * scale * z.shape[0]
    primal = torch.linalg.solve(
        z.t() @ z + regularization * torch.eye(z.shape[1]), z.t()
    )
    assert torch.allclose(projection.t(), primal, atol=2.0e-5, rtol=2.0e-5)


def test_jdrq_hierarchy_refine_is_legal_and_non_worsening() -> None:
    torch.manual_seed(11)
    weight = torch.randn(12, 128) * 0.08
    activation = torch.randn(48, 128) * 0.25
    parent = solution._dense_to_hif4(
        weight,
        search_offsets=solution._WEIGHT_OFFSETS,
        error_threshold=solution._WEIGHT_REFINE_ERROR_THRESHOLD,
        accept_margin=solution._WEIGHT_REFINE_ACCEPT_MARGIN,
        max_refine_ratio=1.0,
        max_refine_blocks=solution._WEIGHT_REFINE_MAX_BLOCKS,
    )
    teacher = activation @ weight.t()
    refined = solution._jdrq_refine_hierarchy_offsets(
        activation,
        teacher,
        weight,
        parent,
        max_ratio=1.0,
        max_blocks=2,
        offsets=solution._JDRQ_HIERARCHY_OFFSETS,
    )
    parent_loss = solution._jdrq_product_loss(activation, teacher, parent)
    refined_loss = solution._jdrq_product_loss(activation, teacher, refined)
    assert refined_loss <= parent_loss + 1.0e-7
    assert set(refined) == {
        "scale_factor",
        "scale_lv2",
        "scale_lv3",
        "sign",
        "mant",
    }
    for key, value in refined.items():
        assert torch.isfinite(value).all(), key
    assert torch.all((refined["mant"] * 4.0 >= 0.0) & (refined["mant"] * 4.0 <= 7.0))
    assert torch.all((refined["scale_lv2"] == 1.0) | (refined["scale_lv2"] == 2.0))
    assert torch.all((refined["scale_lv3"] == 1.0) | (refined["scale_lv3"] == 2.0))


def test_jdrq_rowwise_hierarchy_is_legal_and_non_worsening() -> None:
    """C75.3 row-specific block budgets keep the parent fallback valid."""

    torch.manual_seed(13)
    weight = torch.randn(16, 192) * 0.08
    activation = torch.randn(64, 192) * 0.25
    parent = solution._dense_to_hif4(
        weight,
        search_offsets=solution._WEIGHT_OFFSETS,
        error_threshold=solution._WEIGHT_REFINE_ERROR_THRESHOLD,
        accept_margin=solution._WEIGHT_REFINE_ACCEPT_MARGIN,
        max_refine_ratio=1.0,
        max_refine_blocks=solution._WEIGHT_REFINE_MAX_BLOCKS,
    )
    teacher = activation @ weight.t()
    refined = solution._jdrq_refine_rowwise_hierarchy(
        activation,
        teacher,
        weight,
        parent,
        max_ratio=0.25,
        max_blocks=2,
        offsets=solution._JDRQ_HIERARCHY_OFFSETS,
    )
    parent_loss = solution._jdrq_product_loss(activation, teacher, parent)
    refined_loss = solution._jdrq_product_loss(activation, teacher, refined)
    assert refined_loss <= parent_loss + 1.0e-7
    assert set(refined) == set(parent)
    for key, value in refined.items():
        assert torch.isfinite(value).all(), key


def test_jdrq_product_builder_does_not_mutate_activation_state() -> None:
    torch.manual_seed(19)
    exact = torch.randn(12, 64) * 0.2
    pair = nvfp4_encode(exact, "amax6")
    state = {
        "smooth_inv": None,
        "permutation": None,
        "block_smooth_size": 0,
        "block_smooth_seed": 0,
        "cat_transform": None,
        "importance": torch.ones(64),
        "gram": None,
        "gram8": None,
        "gram16": None,
        "offsets": torch.tensor((0,), dtype=torch.int8),
        "error_threshold": 0.0,
        "accept_margin": 0.0,
        "max_refine_ratio": 0.0,
        "max_refine_blocks": 0,
        "in_features": 64,
    }
    before = {
        key: (value.clone() if torch.is_tensor(value) else value)
        for key, value in state.items()
    }
    d = torch.ones(64)
    permutation = solution._identity_permutation(64, torch.device("cpu"))
    solution._jdrq_calibration_products(
        [pair], state, d, permutation, 0, 0, None, max_rows=12
    )
    for key, value in before.items():
        if torch.is_tensor(value):
            assert torch.equal(state[key], value), key
        else:
            assert state[key] == value, key


def test_source_scale_proposals_are_legal_and_non_worsening() -> None:
    """C75 source proposals extend the pool without replacing the parent."""

    torch.manual_seed(23)
    dense = torch.randn(12, 128) * 0.35
    pair = nvfp4_encode(dense, "amax6")
    codes = solution._source_scale_code_candidates(
        pair[1], dense.shape, dense.shape[-1]
    )
    assert codes is not None
    assert tuple(codes.shape) == (12, 2, 3)
    assert torch.isfinite(codes.to(torch.float32)).all()
    assert bool(((codes >= 0) & (codes <= 254)).all())

    flat_gram = torch.eye(4).repeat(dense.shape[-1] // 4, 1, 1)
    common = dict(
        group_gram=flat_gram,
        search_offsets=(-1, 1, 2, 3),
        error_threshold=0.0,
        accept_margin=0.0,
        max_refine_ratio=1.0,
        max_refine_blocks=10_000,
    )
    parent = solution._nvfp4_to_hif4(
        *pair, source_scale_proposal=False, **common
    )
    source = solution._nvfp4_to_hif4(
        *pair, source_scale_proposal=True, **common
    )
    reference = solution._dequantize_nvfp4_float32(*pair)
    parent_loss = (solution._dequantize_hif4(parent) - reference).square().mean()
    source_loss = (solution._dequantize_hif4(source) - reference).square().mean()
    assert source_loss <= parent_loss + 1.0e-6
    assert set(source) == set(parent)


def test_activation_gram64_refine_is_non_worsening() -> None:
    """C75.2 full-64 activation metric keeps the legal parent fallback."""

    torch.manual_seed(29)
    dense = torch.randn(10, 128) * 0.3
    pair = nvfp4_encode(dense, "amax6")
    params = solution._dense_to_hif4(
        solution._dequantize_nvfp4_float32(*pair),
        search_offsets=(-1, 1, 2, 3),
        max_refine_ratio=1.0,
        max_refine_blocks=10_000,
    )
    gram64 = torch.eye(64).repeat(2, 1, 1)
    refined = solution._refine_activation_blocks64(
        solution._dequantize_nvfp4_float32(*pair), params, gram64,
        max_ratio=1.0, max_blocks=2, sweeps=1,
    )
    reference = solution._dequantize_nvfp4_float32(*pair)
    parent_error = solution._dequantize_hif4(params) - reference
    refined_error = solution._dequantize_hif4(refined) - reference
    assert refined_error.square().mean() <= parent_error.square().mean() + 1.0e-6
    assert set(refined) == set(params)
    for key, value in refined.items():
        assert torch.isfinite(value).all(), key


def test_activation_gram64_hierarchy_is_non_worsening() -> None:
    """C75.5 scale/lv2/lv3 beam keeps the full-H parent fallback."""

    torch.manual_seed(31)
    dense = torch.randn(8, 128) * 0.3
    params = solution._dense_to_hif4(
        dense,
        search_offsets=(-1, 1, 2, 3),
        max_refine_ratio=1.0,
        max_refine_blocks=10_000,
    )
    gram64 = torch.stack(
        [
            dense[:, lo : lo + 64].t().mm(dense[:, lo : lo + 64])
            / float(dense.shape[0])
            + 0.05 * torch.eye(64)
            for lo in (0, 64)
        ],
        dim=0,
    )
    refined = solution._refine_activation_hierarchy64(
        dense,
        params,
        gram64,
        max_ratio=1.0,
        max_blocks=2,
        offsets=solution._ACTIVATION_GRAM64_HIERARCHY_OFFSETS,
        accept_margin=0.0,
    )
    reference = dense.reshape(dense.shape[0], 2, 64)
    parent_error = solution._dequantize_hif4(params).reshape_as(reference) - reference
    refined_error = solution._dequantize_hif4(refined).reshape_as(reference) - reference
    parent_loss = torch.einsum("rbi,bij,rbj->", parent_error, gram64, parent_error)
    refined_loss = torch.einsum("rbi,bij,rbj->", refined_error, gram64, refined_error)
    assert refined_loss <= parent_loss + 1.0e-6
    assert set(refined) == set(params)
    for key, value in refined.items():
        assert torch.isfinite(value).all(), key
