from __future__ import annotations

import torch

import solution


def test_global_activation_lrh_state_is_finite_and_bounded() -> None:
    torch.manual_seed(703)
    weight = torch.randn(96, 128) * 0.05
    gram64 = solution._gram64(weight)
    lowrank = solution._global_activation_lrh(weight, gram64)
    assert lowrank is not None
    assert lowrank.shape[0] == 128
    assert lowrank.shape[1] <= solution._ACT_GLOBAL_LRH_RANK
    assert torch.isfinite(lowrank).all()


def test_global_activation_lrh_gate_uses_exact_deployed_gram() -> None:
    torch.manual_seed(704)
    dense = torch.randn(7, 128) * 0.1
    weight = torch.randn(96, 128) * 0.05
    gram64 = solution._gram64(weight)
    parent = solution._dense_to_hif4(dense, gram64=gram64)
    deployed = solution._dequantize_hif4(parent).to(torch.float32)
    deployment_gram = deployed.t().mm(deployed)
    lowrank = solution._global_activation_lrh(
        deployed, solution._gram64(deployed)
    )
    assert lowrank is not None
    refined, diagnostics = solution._refine_activation_global_lrh(
        dense,
        parent,
        gram64,
        deployment_gram,
        lowrank,
        return_diagnostics=True,
    )
    assert set(refined) == set(parent)
    assert diagnostics["proposal_rows"] >= diagnostics["accepted_rows"]
    assert diagnostics["accepted_rows"] <= diagnostics["gram_accept_rows"]
    before = solution._dequantize_hif4(parent).to(torch.float32) - dense
    after = solution._dequantize_hif4(refined).to(torch.float32) - dense
    before_loss = (before.mm(deployment_gram) * before).sum(dim=1)
    after_loss = (after.mm(deployment_gram) * after).sum(dim=1)
    assert torch.all(after_loss <= before_loss + 1.0e-5)


def test_global_activation_lrh_skips_wide_inputs() -> None:
    torch.manual_seed(705)
    weight = torch.randn(64, 2048) * 0.05
    assert solution._global_activation_lrh(weight, solution._gram64(weight)) is None
