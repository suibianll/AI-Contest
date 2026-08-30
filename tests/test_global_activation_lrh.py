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


def test_global_activation_lrh_wide_rank4_is_bounded_and_finite() -> None:
    """L6b accepts only the explicit wide-shape rank-4 path.

    A zero block-diagonal operand is sufficient for this synthetic range
    check and avoids materializing a second dense Gram matrix in the test.
    """

    torch.manual_seed(708)
    for channels in (2048, 4096, 4864, 8192):
        rows = 8
        weight = torch.randn(rows, channels) * 0.05
        blocks = channels // solution._BLOCK
        gram64 = torch.zeros(blocks, solution._BLOCK, solution._BLOCK)
        lowrank = solution._global_activation_lrh(
            weight,
            gram64,
            rank=solution._ACT_GLOBAL_LRH_WIDE_RANK,
            max_channels=solution._ACT_GLOBAL_LRH_WIDE_MAX_CHANNELS,
        )
        assert lowrank is not None
        assert tuple(lowrank.shape) == (channels, solution._ACT_GLOBAL_LRH_WIDE_RANK)
        assert torch.isfinite(lowrank).all()


def test_global_activation_lrh_wide_path_respects_channel_cap() -> None:
    torch.manual_seed(709)
    channels = 2048
    weight = torch.randn(8, channels) * 0.05
    gram64 = torch.zeros(channels // solution._BLOCK, solution._BLOCK, solution._BLOCK)
    assert solution._global_activation_lrh(
        weight,
        gram64,
        rank=solution._ACT_GLOBAL_LRH_WIDE_RANK,
        max_channels=channels - solution._BLOCK,
    ) is None


def test_g64_hierarchy_sweep_matches_independent_coordinate_bruteforce() -> None:
    """The fixed-scale hierarchy sweep must agree with an exhaustive reference."""

    torch.manual_seed(710)
    dense = torch.randn(1, 64) * 0.12
    gram64 = solution._gram64(torch.randn(96, 64) * 0.05)
    parent = solution._dense_to_hif4(dense, offsets=(0,), gram64=gram64)
    refined, diagnostics = solution._refine_activation_hierarchy_g64(
        dense, parent, gram64, max_blocks=1, sweeps=1, return_diagnostics=True
    )
    assert diagnostics["selected_blocks"] == 1

    x = dense.reshape(8, 2, 4).to(torch.float32)
    sf = parent["scale_factor"].reshape(1, 1, 1, 1, 1)[0, 0, 0, 0, 0]
    lv2 = parent["scale_lv2"].reshape(1, 1, 8, 1, 1)[0, 0, :, 0, 0].clone()
    lv3 = parent["scale_lv3"].reshape(1, 1, 8, 2, 1)[0, 0, :, :, 0].clone()
    q = solution._dequantize_hif4(parent).reshape(64).to(torch.float32)
    sign = parent["sign"].reshape(1, 1, 8, 2, 4)[0, 0].clone()
    mant = parent["mant"].reshape(1, 1, 8, 2, 4)[0, 0].clone()
    gram = gram64[0].to(torch.float32)

    def encode(trial_lv2: torch.Tensor, trial_lv3: torch.Tensor):
        denominator = sf * trial_lv2[:, None, None] * trial_lv3[:, :, None]
        trial_mant = torch.round(x.abs() * 4.0 / denominator).clamp(0.0, 7.0) * 0.25
        trial_sign = torch.sign(x)
        trial_sign = torch.where(trial_mant == 0.0, torch.zeros_like(trial_sign), trial_sign)
        return (trial_sign * trial_mant * denominator).reshape(64), trial_sign, trial_mant

    for group in range(8):
        current = float(lv2[group].item())
        best = None
        for value in (1.0, 2.0):
            if abs(current - value) <= solution._EPS:
                continue
            trial_lv2 = lv2.clone()
            trial_lv2[group] = value
            trial_q, trial_sign, trial_mant = encode(trial_lv2, lv3)
            step = trial_q - q
            gram_step = gram.mv(step)
            delta = 2.0 * torch.dot(q - dense.reshape(64), gram_step) + torch.dot(step, gram_step)
            if torch.isfinite(delta) and delta < 0:
                if best is None or float(delta) < best[0]:
                    best = (float(delta), value, trial_q, trial_sign, trial_mant)
        if best is not None:
            _, value, q, sign, mant = best
            lv2[group] = value

    for group in range(8):
        for subgroup in range(2):
            current = float(lv3[group, subgroup].item())
            best = None
            for value in (1.0, 2.0):
                if abs(current - value) <= solution._EPS:
                    continue
                trial_lv3 = lv3.clone()
                trial_lv3[group, subgroup] = value
                trial_q, trial_sign, trial_mant = encode(lv2, trial_lv3)
                step = trial_q - q
                gram_step = gram.mv(step)
                delta = 2.0 * torch.dot(q - dense.reshape(64), gram_step) + torch.dot(step, gram_step)
                if torch.isfinite(delta) and delta < 0:
                    if best is None or float(delta) < best[0]:
                        best = (float(delta), value, trial_q, trial_sign, trial_mant)
            if best is not None:
                _, value, q, sign, mant = best
                lv3[group, subgroup] = value

    expected = solution._dequantize_hif4(refined).reshape(64).to(torch.float32)
    torch.testing.assert_close(expected, q, rtol=0.0, atol=1.0e-7)
    torch.testing.assert_close(
        refined["scale_lv2"].reshape(1, 1, 8, 1, 1)[0, 0, :, 0, 0], lv2
    )
    torch.testing.assert_close(
        refined["scale_lv3"].reshape(1, 1, 8, 2, 1)[0, 0, :, :, 0], lv3
    )


def test_final_gram_selector_is_rowwise_nonincreasing() -> None:
    torch.manual_seed(706)
    dense = torch.randn(5, 128) * 0.1
    parent_gram = solution._gram64(torch.randn(64, 128) * 0.05)
    deployment = torch.randn(64, 128) * 0.05
    deployment_gram = deployment.t().mm(deployment)
    parent = solution._dense_to_hif4(dense, gram64=parent_gram)
    candidate = solution._dense_to_hif4(
        dense, offsets=(-4, -2, 0, 2, 4), gram64=solution._gram64(deployment)
    )
    selected = solution._select_activation_by_deployment_gram(
        dense, parent, candidate, deployment_gram
    )
    before = solution._dequantize_hif4(parent).to(torch.float32) - dense
    after = solution._dequantize_hif4(selected).to(torch.float32) - dense
    before_loss = (before.mm(deployment_gram) * before).sum(dim=1)
    after_loss = (after.mm(deployment_gram) * after).sum(dim=1)
    assert torch.all(after_loss <= before_loss + 1.0e-5)


def test_final_gram_gals_exact_gate_is_nonincreasing() -> None:
    torch.manual_seed(707)
    dense = torch.randn(3, 128) * 0.1
    deployment = torch.randn(64, 128) * 0.05
    gram64 = solution._gram64(deployment)
    deployment_gram = deployment.t().mm(deployment)
    parent = solution._dense_to_hif4(dense, gram64=gram64)
    refined, gain = solution._refine_activation_gals_final(
        dense,
        parent,
        gram64,
        max_blocks=2,
        deployment_gram=deployment_gram,
        return_gain=True,
    )
    before = solution._dequantize_hif4(parent).to(torch.float32) - dense
    after = solution._dequantize_hif4(refined).to(torch.float32) - dense
    before_loss = (before.mm(deployment_gram) * before).sum(dim=1)
    after_loss = (after.mm(deployment_gram) * after).sum(dim=1)
    assert torch.all(after_loss <= before_loss + 1.0e-5)
    assert gain >= -1.0e-5
