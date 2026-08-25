from __future__ import annotations

import pytest
import torch

from hif4_system.scoring import attention_output, competition_score


def test_competition_score_matches_rule() -> None:
    reference = torch.tensor([0.0, 2.0])
    standard = torch.tensor([1.0, 3.0])
    player = torch.tensor([0.5, 2.5])

    assert competition_score(reference, standard, player) == pytest.approx(0.75)


@pytest.mark.parametrize("causal", [False, True])
def test_gqa_attention_output_shape(causal: bool) -> None:
    output = attention_output(
        torch.randn(7, 256),
        torch.randn(7, 128),
        torch.randn(7, 128),
        q_num_heads=4,
        kv_num_heads=2,
        head_dim=64,
        causal=causal,
    )

    assert output.shape == (7, 256)
    assert torch.isfinite(output).all()


def test_causal_attention_does_not_read_future_values() -> None:
    q = torch.ones(2, 64)
    k = torch.ones(2, 64)
    v_before = torch.zeros(2, 64)
    v_after = v_before.clone()
    v_after[1] = 5.0

    before = attention_output(q, k, v_before, 1, 1, 64, causal=True)
    after = attention_output(q, k, v_after, 1, 1, 64, causal=True)

    assert torch.equal(before[0], after[0])
    assert not torch.equal(before[1], after[1])
