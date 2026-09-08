"""Small correctness checks for the LC0 Linear hardening candidate."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import torch


SOLUTION_PATH = Path(__file__).with_name("solution.py")


def _load_solution():
    spec = importlib.util.spec_from_file_location("lc0_solution", SOLUTION_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {SOLUTION_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _test_explicit_grid_ties(module) -> None:
    scale = torch.tensor([[0.5] * 64], dtype=torch.float32)
    integer_codes = torch.arange(-6, 7, dtype=torch.float32)
    values = (integer_codes + 0.5).mul(0.25 * scale[:, :13])
    values = torch.cat((values, torch.zeros(1, 64 - values.shape[1])), dim=1)

    actual = module._l23_nearest_signed_code(values, scale)
    grid = torch.arange(-7, 8, dtype=torch.float32)
    expected = torch.stack(
        [
            grid[(grid - value).abs().argmin()]
            for value in (values / scale / 0.25).reshape(-1)
        ]
    ).reshape_as(actual)
    assert torch.equal(actual, expected), "L23 projection tie rule diverges from grid"


def _test_strict_gptq_failure(module) -> None:
    state = {
        "gptq_block_order": torch.tensor([0, 0], dtype=torch.int16),
        "in_features": 128,
        "smooth_inv": None,
        "permutation": None,
        "block_smooth_size": 0,
        "residual_u": None,
        "residual_v": None,
        "rank1_u": torch.zeros(128),
        "rank1_v": torch.zeros(128),
    }
    quant = torch.zeros(1, 128)
    scale = torch.ones(1, 8)
    try:
        module.hif4_dynamic_quantize_activation(quant, scale, state)
    except RuntimeError as exc:
        assert "refusing silent fallback" in str(exc)
    else:
        raise AssertionError("invalid GPTQ state silently fell back")


def _test_strict_shape_failure(module) -> None:
    weight_quant = torch.zeros(1, 128)
    weight_scale = torch.ones(1, 8)
    weight_dense = module._dequantize_nvfp4_float32(weight_quant, weight_scale)
    weight_params = module._ref_encode_standard_hif4(weight_dense)
    wrong_activation_params = module._ref_encode_standard_hif4(
        torch.zeros(1, 128)
    )
    original_dynamic = module.hif4_dynamic_quantize_activation

    def wrong_dynamic(*args, **kwargs):
        return wrong_activation_params

    module.hif4_dynamic_quantize_activation = wrong_dynamic
    try:
        module._l23_residual_subspace_fit(
            weight_params,
            {},
            weight_quant,
            weight_scale,
            [(torch.zeros(2, 128), torch.ones(2, 8))],
        )
    except RuntimeError as exc:
        assert "shape mismatch" in str(exc)
    else:
        raise AssertionError("activation shape mismatch was silently reshaped")
    finally:
        module.hif4_dynamic_quantize_activation = original_dynamic


def _test_calibration_replay_and_canonical_zero(module) -> None:
    torch.manual_seed(7)
    torch.set_num_threads(1)
    weight_quant = torch.randn(2, 128).clamp(-1.5, 1.5)
    weight_scale = torch.full((2, 8), 0.5)
    calibration = [
        (torch.randn(24, 128).clamp(-1.5, 1.5), torch.full((24, 8), 0.5))
        for _ in range(2)
    ]
    result = module.hif4_calibration_and_quantize_weight(
        weight_quant, weight_scale, calibration
    )
    params = result["weight_params"]
    state = result["activation_state"]
    assert int(((params["mant"] == 0) & (params["sign"] != 0)).sum()) == 0
    assert int((params["mant"] == 0).sum()) > 0
    decoded = module._dequantize_hif4(params)
    explicit = (
        params["sign"]
        * params["mant"]
        * params["scale_lv3"]
        * params["scale_lv2"]
        * params["scale_factor"]
    ).flatten(start_dim=-4, end_dim=-1)
    assert decoded.shape == weight_quant.shape
    assert torch.equal(decoded, explicit)
    for quant, scale in calibration:
        first = module.hif4_dynamic_quantize_activation(quant, scale, state)
        second = module.hif4_dynamic_quantize_activation(quant, scale, state)
        assert all(torch.equal(first[key], second[key]) for key in first)
        assert module._dequantize_hif4(first).shape == quant.shape
        assert int(((first["mant"] == 0) & (first["sign"] != 0)).sum()) == 0


def _test_official_objective_formula() -> None:
    residuals = [torch.randn(3, 5), torch.randn(2, 5)]
    mse_standard = [0.25, 2.0]
    folds = len(residuals)
    weighted_sum = sum(
        residual.square().sum()
        / (folds * residual.numel() * denominator)
        for residual, denominator in zip(residuals, mse_standard)
    )
    direct_mean = sum(
        residual.square().mean() / denominator
        for residual, denominator in zip(residuals, mse_standard)
    ) / folds
    assert torch.allclose(
        weighted_sum,
        direct_mean,
        rtol=0.0,
        atol=1.0e-7,
    )


def _test_incremental_residual_formula() -> None:
    torch.manual_seed(23)
    x_block = torch.randn(7, 64)
    w_old = torch.randn(4, 64)
    w_new = torch.randn(4, 64)
    old_residual = torch.randn(7, 4)
    target = x_block @ w_old.t() + old_residual
    incremental = old_residual + x_block @ (w_old - w_new).t()
    full = target - x_block @ w_new.t()
    assert torch.allclose(incremental, full, rtol=0.0, atol=1.0e-5)


def main() -> None:
    module = _load_solution()
    _test_explicit_grid_ties(module)
    _test_strict_gptq_failure(module)
    _test_strict_shape_failure(module)
    _test_calibration_replay_and_canonical_zero(module)
    _test_official_objective_formula()
    _test_incremental_residual_formula()
    print("LC0 correctness battery: PASS")


if __name__ == "__main__":
    main()
