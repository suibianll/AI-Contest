from __future__ import annotations

import torch

import solution
from evaluator.linear_ceiling_dashboard import (
    ALL_SCALE_OFFSETS,
    _all_code_oracle_loss,
    _block_loss,
)


def test_vectorized_scale_oracle_matches_reference_plain_and_gram() -> None:
    torch.manual_seed(7)
    dense = torch.randn(3, 128)
    gram = solution._gram64(torch.randn(5, 128))

    for metric in (None, gram):
        vectorized = _all_code_oracle_loss(solution, dense, metric)
        exhaustive_params = solution._encode_rows(
            dense, ALL_SCALE_OFFSETS, gram64=metric
        )
        exhaustive = _block_loss(solution, dense, exhaustive_params, metric)
        torch.testing.assert_close(vectorized, exhaustive, rtol=1.0e-7, atol=3.0e-7)
