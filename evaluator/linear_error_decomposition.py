"""Operand-local Linear error decomposition (evaluator-side, read-only).

This tool reports compliance-legal operand diagnostics:

- ``activation_local_error``  — mean squared error of the quantized
  activation operand (never the Linear output);
- ``activation_tail_cvar``    — CVaR (worst-quartile mean) of the
  per-row squared activation error;
- ``weight_plain_error``      — mean squared error of the quantized
  weight operand;
- ``weight_hessian_error``    — the Gram-weighted weight error
  ``tr(W_err G W_err^T) / (M*N)`` with ``G = A^T A / N``, i.e. the
  Hessian loss that ``Q(W)`` is already allowed to consume;
- ``transform_orthogonality_error`` — Frobenius deviation ``T^T T - I``
  of the composed calibration transform.

The tool never forms the reference Linear output ``A @ W^T``, never
forms the cross residual, and never returns anything with the shape
``[tokens, out_features]``; outputs are scalars or per-row/per-channel
1-D tensors only, so nothing here can be fed back into ``Q(A)``.
"""

from __future__ import annotations

import torch

__all__ = ["decompose_linear_error"]


def _row_squared_error(reference: torch.Tensor, quantized: torch.Tensor) -> torch.Tensor:
    delta = quantized.to(torch.float32) - reference.to(torch.float32)
    return delta.square().sum(dim=-1)


def _tail_cvar(row_error: torch.Tensor, quartile: float = 0.25) -> float:
    if row_error.numel() == 0:
        return 0.0
    count = max(1, int(round(float(row_error.numel()) * quartile)))
    worst = torch.topk(row_error, k=count, largest=True).values
    return float(worst.mean())


def _transform_orthogonality_error(
    transform: torch.Tensor | None,
) -> float:
    if transform is None:
        return 0.0
    t = transform.detach().to(torch.float32)
    gram = t.transpose(-2, -1) @ t
    identity = torch.eye(
        gram.shape[-1], dtype=gram.dtype, device=gram.device
    ).expand_as(gram)
    return float((gram - identity).square().sum(dim=(-2, -1)).mean())


def decompose_linear_error(
    activation_reference: torch.Tensor,
    activation_quantized: torch.Tensor,
    weight_reference: torch.Tensor,
    weight_quantized: torch.Tensor,
    *,
    transform: torch.Tensor | None = None,
    tail_quartile: float = 0.25,
) -> dict[str, torch.Tensor | float]:
    """Decompose operand-local Linear error contributions.

    Args:
        activation_reference: ``[N, K]`` dense reference activation operand.
        activation_quantized: ``[N, K]`` player-quantized activation operand.
        weight_reference: ``[M, K]`` dense reference weight operand.
        weight_quantized: ``[M, K]`` player-quantized weight operand.
        transform: optional ``[K, K]`` composed calibration transform used
            to report orthogonality error (permutation / block Hadamard /
            smooth scaling composed as a single matrix).
        tail_quartile: fraction of worst rows averaged by the tail CVaR.

    Returns:
        Dict with the five operand-local diagnostics from the module
        docstring plus the per-row activation error tensor.  No Linear
        output, cross residual, or ``[N, M]`` tensor is ever produced.
    """

    activation_reference = activation_reference.detach().to(torch.float32)
    activation_quantized = activation_quantized.detach().to(torch.float32)
    weight_reference = weight_reference.detach().to(torch.float32)
    weight_quantized = weight_quantized.detach().to(torch.float32)

    if activation_reference.shape != activation_quantized.shape:
        raise ValueError("activation operands must share one shape")
    if weight_reference.shape != weight_quantized.shape:
        raise ValueError("weight operands must share one shape")
    if activation_reference.shape[-1] != weight_reference.shape[-1]:
        raise ValueError("activation/weight inner dimensions must match")

    rows, channels = activation_reference.shape[-2], activation_reference.shape[-1]
    out_features = weight_reference.shape[-2]

    activation_row_error = _row_squared_error(
        activation_reference, activation_quantized
    )
    activation_local_error = float(activation_row_error.mean())

    weight_delta = weight_quantized - weight_reference
    weight_plain_error = float(
        weight_delta.square().sum(dim=-1).mean()
    )

    # Hessian loss: tr(W_err (A^T A / N) W_err^T) / (M*N).  This is the
    # only quantity that combines both operands; it is the exact loss
    # Q(W) already consumes and is therefore compliance-legal.
    gram = (
        activation_reference.transpose(-2, -1) @ activation_reference
    ) / float(rows)
    weight_hessian_error = float(
        torch.einsum(
            "...mk,...kn,...mn->...", weight_delta, gram, weight_delta
        ).sum(dim=-1).mean()
        / float(out_features)
    )

    return {
        "activation_local_error": activation_local_error,
        "activation_tail_cvar": _tail_cvar(
            activation_row_error, quartile=tail_quartile
        ),
        "weight_hessian_error": weight_hessian_error,
        "weight_plain_error": weight_plain_error,
        "transform_orthogonality_error": _transform_orthogonality_error(
            transform
        ),
        "activation_row_error": activation_row_error,
    }
