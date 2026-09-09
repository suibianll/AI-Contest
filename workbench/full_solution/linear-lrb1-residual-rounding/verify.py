"""Focused L-RB1 legality, parent-control and reachability check."""

from pathlib import Path
import importlib.util
import sys

import torch


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]


def load_solution(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def assert_params(params: dict[str, torch.Tensor], rows: int, channels: int) -> None:
    expected = {
        "scale_factor": (rows, channels // 64, 1, 1, 1),
        "scale_lv2": (rows, channels // 64, 8, 1, 1),
        "scale_lv3": (rows, channels // 64, 8, 2, 1),
        "sign": (rows, channels // 64, 8, 2, 4),
        "mant": (rows, channels // 64, 8, 2, 4),
    }
    if set(params) != set(expected):
        raise AssertionError(f"unexpected HiF4 keys: {set(params)}")
    for key, shape in expected.items():
        if tuple(params[key].shape) != shape:
            raise AssertionError(f"{key} shape {tuple(params[key].shape)} != {shape}")


def deployment_view(candidate, weight_quant, weight_scale, state, params, rows, channels):
    """Return (floor_code, code_abs, sign_field, frac, eligible) for a parent state."""

    target = candidate._lrb1_reconstruct_deployment_weight(
        weight_quant, weight_scale, state
    )
    if target is None:
        raise AssertionError("deployment weight reconstruction failed")
    blocks = channels // 64
    scale = params["scale_factor"].to(torch.float32).reshape(rows, blocks, 1, 1, 1)
    lv2 = params["scale_lv2"].to(torch.float32).reshape(rows, blocks, 8, 1, 1)
    lv3 = params["scale_lv3"].to(torch.float32).reshape(rows, blocks, 8, 2, 1)
    denominator = (scale * lv2 * lv3).repeat_interleave(4, dim=-1).reshape(rows, channels)
    sign_field = params["sign"].to(torch.float32).reshape(rows, channels)
    code_abs = torch.round(
        params["mant"].to(torch.float32).reshape(rows, channels) / 0.25
    ).clamp_(0.0, 7.0)
    magnitude = 4.0 * target.abs() / denominator.clamp_min(1.0e-12)
    nearest = torch.round(magnitude).clamp(0.0, 7.0)
    floor_code = torch.floor(magnitude)
    frac = (magnitude - floor_code).clamp(0.0, 1.0)
    eligible = (
        (code_abs == nearest) & (floor_code >= 1.0) & (floor_code <= 6.0)
    )
    return floor_code, code_abs, sign_field, frac, eligible


def main() -> None:
    candidate = load_solution(HERE / "candidate" / "solution.py", "lrb1_candidate")
    parent = load_solution(ROOT / "solution.py", "lrb1_parent")
    required = (
        "hif4_calibration_and_quantize_weight",
        "hif4_dynamic_quantize_activation",
        "hif4_calibration_attention",
        "hif4_dynamic_quantize_q",
        "hif4_dynamic_quantize_k",
        "hif4_dynamic_quantize_v",
    )
    missing = [name for name in required if not hasattr(candidate, name)]
    if missing:
        raise AssertionError(f"missing public API: {missing}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    generator = torch.Generator(device=device).manual_seed(226)
    rows, channels = 64, 128
    weight_quant = torch.randn((rows, channels), generator=generator, device=device)
    weight_scale = torch.ones((rows, 8), dtype=torch.float32, device=device)
    activations = [
        (
            torch.randn((12, channels), generator=generator, device=device),
            torch.ones((12, 8), dtype=torch.float32, device=device),
        ),
        (
            torch.randn((10, channels), generator=generator, device=device),
            torch.ones((10, 8), dtype=torch.float32, device=device),
        ),
    ]

    parent_result = parent.hif4_calibration_and_quantize_weight(
        weight_quant, weight_scale, activations
    )
    result = candidate.hif4_calibration_and_quantize_weight(
        weight_quant, weight_scale, activations
    )
    if set(result) != {"weight_params", "activation_state"}:
        raise AssertionError(f"unexpected calibration result keys: {set(result)}")
    assert_params(result["weight_params"], rows, channels)
    assert_params(parent_result["weight_params"], rows, channels)
    state = result["activation_state"]
    if state.get("lrb1_attempted") != 1:
        raise AssertionError("L-RB1 did not reach its fixed boundary fit")
    if state.get("lrb1_coordinate") != (
        "deployment-output-shared-weight-rounding-boundary"
    ):
        raise AssertionError("L-RB1 coordinate provenance is missing")
    if int(state.get("lrb1_boundary_count", 0)) != 12:
        raise AssertionError("L-RB1 boundary count is not the fixed 12")
    if int(state.get("lrb1_fit_windows", 0)) != 2:
        raise AssertionError("L-RB1 did not consume every supplied window")
    if int(state.get("lrb1_eligible", 0)) <= 0:
        raise AssertionError("L-RB1 found no eligible deployment-coordinate element")

    activation_params = candidate.hif4_dynamic_quantize_activation(
        activations[0][0], activations[0][1], state
    )
    assert_params(activation_params, 12, channels)

    floor_code, code_abs, sign_field, frac, eligible = deployment_view(
        candidate,
        weight_quant,
        weight_scale,
        state,
        parent_result["weight_params"],
        rows,
        channels,
    )

    # Control A: the parent boundary table must restore the parent codes bit
    # for bit and must not change a single mantissa.
    parent_table = torch.full((2, 8), 0.5, device=device, dtype=torch.float32)
    restored, restored_switched = candidate._lrb1_apply_boundaries(
        floor_code, code_abs, sign_field, frac, eligible, parent_table
    )
    restored_changed = restored_switched & (restored != code_abs)
    if int(restored_changed.sum()) != 0:
        raise AssertionError("parent boundary 0.5 changed an eligible mantissa")
    if not bool(torch.equal(restored, code_abs)):
        raise AssertionError("parent boundary 0.5 did not restore the parent codes")

    # Control B: a synthetic non-parent boundary must switch exactly the
    # eligible elements whose fractional part crosses it.
    synthetic = parent_table.clone()
    synthetic[1, 3] = 0.25
    synth_code, synth_switched = candidate._lrb1_apply_boundaries(
        floor_code, code_abs, sign_field, frac, eligible, synthetic
    )
    expected = eligible & (sign_field == 1.0) & (floor_code == 3.0) & (frac >= 0.25)
    expected_count = int(expected.sum())
    if expected_count == 0:
        raise AssertionError("synthetic boundary has no reachable eligible element")
    synth_class = synth_switched & (sign_field == 1.0) & (floor_code == 3.0)
    if int(synth_class.sum()) != expected_count:
        raise AssertionError(
            f"synthetic boundary switched {int(synth_class.sum())} != {expected_count}"
        )
    if not bool(torch.all(synth_code[expected] == 4.0)):
        raise AssertionError("synthetic boundary did not raise the target code")

    # Arm consistency: a rejected table must return the parent five fields
    # bit for bit; an accepted table must differ only by the recorded +0.25
    # mantissa steps.
    arm = state.get("lrb1_arm")
    changed_count = int(state.get("lrb1_changed_mantissa", 0))
    parent_mant = parent_result["weight_params"]["mant"]
    candidate_mant = result["weight_params"]["mant"]
    if arm == "parent":
        if changed_count != 0:
            raise AssertionError("rejected table still reports changed mantissas")
        for key, value in parent_result["weight_params"].items():
            if not bool(torch.equal(value, result["weight_params"][key])):
                raise AssertionError(f"rejected table changed field {key}")
    elif arm == "accepted":
        if changed_count <= 0:
            raise AssertionError("accepted table reports no changed mantissa")
        delta = (candidate_mant.to(torch.float32) - parent_mant.to(torch.float32))
        if not bool(torch.all((delta == 0.0) | (delta == 0.25))):
            raise AssertionError("accepted table produced a non +0.25 mantissa step")
        if int((delta != 0.0).sum()) != changed_count:
            raise AssertionError("changed count disagrees with the mantissa delta")
        if state.get("lrb1_table_accepted") != 1:
            raise AssertionError("accepted arm did not record table acceptance")
        if not float(state.get("lrb1_delta_loss", 0.0)) < 0.0:
            raise AssertionError("accepted table did not strictly decrease loss")
    else:
        raise AssertionError(f"unexpected L-RB1 arm {arm}")

    reference = load_solution(ROOT / "evaluator" / "reference_hif4.py", "lrb1_reference")
    reference.validate_state(state)
    reference.validate_hif4_params(result["weight_params"], (rows, channels))
    reference.validate_hif4_params(activation_params, (12, channels))
    print(
        f"L-RB1 verify passed: device={device.type} arm={arm} "
        f"eligible={state.get('lrb1_eligible')} "
        f"proposals={state.get('lrb1_proposals')} "
        f"changed={changed_count} "
        f"delta_loss={state.get('lrb1_delta_loss')} "
        f"synthetic_switched={expected_count}"
    )


if __name__ == "__main__":
    main()
