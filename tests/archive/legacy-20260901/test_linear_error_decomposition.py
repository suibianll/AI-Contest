from __future__ import annotations

import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "evaluator"))

from linear_error_decomposition import decompose_linear_error  # noqa: E402


def test_exact_operands_report_zero_error() -> None:
    torch.manual_seed(201)
    activation = torch.randn(16, 64)
    weight = torch.randn(8, 64)
    report = decompose_linear_error(
        activation, activation.clone(), weight, weight.clone()
    )
    assert report["activation_local_error"] == 0.0
    assert report["activation_tail_cvar"] == 0.0
    assert report["weight_hessian_error"] == 0.0
    assert report["weight_plain_error"] == 0.0
    assert report["transform_orthogonality_error"] == 0.0


def test_activation_metrics_match_manual_computation() -> None:
    torch.manual_seed(202)
    activation = torch.randn(12, 64)
    quantized = activation + 0.1
    weight = torch.randn(8, 64)
    report = decompose_linear_error(
        activation, quantized, weight, weight.clone()
    )
    row_error = (quantized - activation).square().sum(dim=-1)
    assert abs(report["activation_local_error"] - row_error.mean()) < 1e-9
    worst = torch.topk(row_error, k=3).values
    assert abs(report["activation_tail_cvar"] - worst.mean()) < 1e-9
    assert torch.allclose(report["activation_row_error"], row_error)


def test_weight_hessian_matches_trace_form() -> None:
    torch.manual_seed(203)
    activation = torch.randn(32, 64)
    weight = torch.randn(8, 64)
    weight_quantized = weight + 0.05
    report = decompose_linear_error(
        activation, activation.clone(), weight, weight_quantized
    )
    delta = weight_quantized - weight
    gram = activation.T @ activation / 32.0
    expected = float((delta @ gram @ delta.T).trace() / 8.0)
    assert abs(report["weight_hessian_error"] - expected) < 1e-6
    assert abs(report["weight_plain_error"] - delta.square().sum(-1).mean()) < 1e-9


def test_transform_orthogonality_error_detects_scaling() -> None:
    torch.manual_seed(204)
    identity = torch.eye(64)
    report = decompose_linear_error(
        torch.randn(4, 64),
        torch.randn(4, 64),
        torch.randn(8, 64),
        torch.randn(8, 64),
        transform=identity,
    )
    assert report["transform_orthogonality_error"] == 0.0

    scaled = 2.0 * torch.eye(64)
    report = decompose_linear_error(
        torch.randn(4, 64),
        torch.randn(4, 64),
        torch.randn(8, 64),
        torch.randn(8, 64),
        transform=scaled,
    )
    # ||(2I)^T(2I) - I||_F^2 = ||3I||_F^2 = 9 * 64
    assert abs(report["transform_orthogonality_error"] - 9.0 * 64.0) < 1e-6


def test_report_never_contains_linear_output_shapes() -> None:
    torch.manual_seed(205)
    activation = torch.randn(20, 64)
    weight = torch.randn(10, 64)
    report = decompose_linear_error(
        activation,
        activation + 0.01,
        weight,
        weight + 0.01,
        transform=torch.randn(64, 64),
    )
    for key, value in report.items():
        if torch.is_tensor(value):
            assert value.ndim == 1
            assert value.shape[0] in (20,)
        else:
            assert isinstance(value, float)


def test_mismatched_operand_shapes_are_rejected() -> None:
    torch.manual_seed(206)
    activation = torch.randn(8, 64)
    weight = torch.randn(8, 128)
    try:
        decompose_linear_error(
            activation, activation, weight, weight
        )
    except ValueError:
        pass
    else:  # pragma: no cover
        raise AssertionError("expected ValueError for mismatched inner dims")
