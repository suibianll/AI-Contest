"""Focused L-JRB1 legality, parent-control, threading and reachability check.

Controls:
  A. six public APIs exist and the candidate imports as a single file away
     from the repository;
  B. the parent boundary table (all 0.5) is bit-identical to the parent
     nearest-rounding formula, and a synthetic table switches exactly the
     analytically predicted set;
  C. the learned table is not silently dropped by the v202 block-order
     wrapper, the GPTQ block loop or the refinement solves: a synthetic table
     must change real deployment mantissas on both the initial-only and the
     refined path;
  D. arm consistency: a rejected round returns the parent five fields bit for
     bit; an accepted round changes mantissas by +-0.25 only, and both the
     activation and the weight changed-code counts reproduce independently.
"""

from pathlib import Path
import importlib.util
import shutil
import sys
import tempfile

import torch


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
BOUNDARIES_KEY = "activation_boundaries"


def load_solution(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def assert_params(params: dict, rows: int, channels: int) -> None:
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


def mant_codes(params: dict) -> torch.Tensor:
    return torch.round(
        params["mant"].detach().to(torch.float32) / 0.25
    ).to(torch.int64)


def single_file_import(candidate_path: Path):
    """Import the candidate from a directory with no repository siblings."""

    with tempfile.TemporaryDirectory() as tmp:
        isolated = Path(tmp) / "solution.py"
        shutil.copyfile(candidate_path, isolated)
        module = load_solution(isolated, "jrb1_isolated")
    for name in (
        "hif4_calibration_and_quantize_weight",
        "hif4_dynamic_quantize_activation",
        "hif4_calibration_attention",
        "hif4_dynamic_quantize_q",
        "hif4_dynamic_quantize_k",
        "hif4_dynamic_quantize_v",
    ):
        if not hasattr(module, name):
            raise AssertionError(f"isolated import is missing {name}")
    return module


def control_b(candidate, device: torch.device) -> None:
    """Boundary rule: parent restore + exact analytic synthetic switch."""

    generator = torch.Generator(device=device).manual_seed(227)
    x_abs = torch.rand((4, 3, 8, 2, 4), generator=generator, device=device)
    scale = torch.full(
        (4, 3, 8, 2, 4), 0.375, device=device, dtype=torch.float32
    )
    parent = torch.round(x_abs * (4.0 / scale)).clamp_(0.0, 7.0) * 0.25

    parent_table = torch.full((8,), 0.5, device=device, dtype=torch.float32)
    restored = candidate._jrb1_boundary_mantissa(x_abs, scale, parent_table)
    if not bool(torch.equal(restored, parent)):
        raise AssertionError("all-0.5 table is not bit-identical to the parent")
    if not bool(
        torch.equal(
            candidate._jrb1_boundary_mantissa(x_abs, scale, None), parent
        )
    ):
        raise AssertionError("missing table is not bit-identical to the parent")

    synthetic = parent_table.clone()
    synthetic[3] = 0.25
    switched = candidate._jrb1_boundary_mantissa(x_abs, scale, synthetic)
    u = x_abs * (4.0 / scale)
    floor_code = torch.floor(u)
    frac = u - floor_code
    expected = (floor_code == 3.0) & (frac > 0.25) & (frac < 0.5)
    expected_count = int(expected.sum())
    if expected_count == 0:
        raise AssertionError("synthetic table has no reachable element")
    changed = switched != parent
    if int(changed.sum()) != expected_count:
        raise AssertionError(
            f"synthetic table changed {int(changed.sum())} != {expected_count}"
        )
    if not bool(torch.all(switched[expected] == 1.0)):
        raise AssertionError("synthetic up-switch did not land on code 4")
    if not bool(torch.all(switched[~expected] == parent[~expected])):
        raise AssertionError("synthetic table changed an unexpected element")
    print(f"[B] rule control passed: synthetic switched {expected_count}")


def control_c(
    candidate,
    parent,
    state: dict,
    window: tuple,
    device: torch.device,
) -> None:
    """The table must survive the wrapper, GPTQ loop and refinement solves."""

    parent_params = parent.hif4_dynamic_quantize_activation(
        window[0], window[1], state
    )
    restore_state = dict(state)
    restore_state[BOUNDARIES_KEY] = torch.full(
        (7,), 0.5, device=device, dtype=torch.float32
    )
    restored_params = candidate.hif4_dynamic_quantize_activation(
        window[0], window[1], restore_state
    )
    for key, value in parent_params.items():
        if not bool(torch.equal(value, restored_params[key])):
            raise AssertionError(f"all-0.5 deployment table changed field {key}")

    parent_codes = mant_codes(parent_params)
    readings = {}
    for name, overrides in (
        ("refined", {}),
        ("initial-only", {"max_refine_ratio": 0.0, "max_refine_blocks": 0}),
    ):
        synthetic_state = dict(state)
        synthetic_state.update(overrides)
        synthetic_state[BOUNDARIES_KEY] = torch.full(
            (7,), 0.5, device=device, dtype=torch.float32
        )
        synthetic_state[BOUNDARIES_KEY][3] = 0.25
        synthetic_params = candidate.hif4_dynamic_quantize_activation(
            window[0], window[1], synthetic_state
        )
        changed = int((mant_codes(synthetic_params) != parent_codes).sum())
        readings[name] = changed
        if changed <= 0:
            raise AssertionError(
                f"synthetic table was dropped on the {name} path"
            )
    print(
        f"[C] threading control passed: synthetic changed codes "
        f"refined={readings['refined']} initial-only={readings['initial-only']}"
    )


def control_d(
    candidate,
    parent,
    result: dict,
    parent_result: dict,
    state: dict,
    activations: list,
    rows: int,
    channels: int,
    device: torch.device,
) -> str:
    """Arm consistency, independent recounts and payload legality."""

    arm = str(state.get("jrb1_arm"))
    parent_params = parent_result["weight_params"]
    candidate_params = result["weight_params"]
    changed_reported = int(state.get("jrb1_w_changed_mantissa", 0))
    act_reported = int(state.get("jrb1_act_changed_mantissa", 0))

    if arm == "joint":
        delta = candidate_params["mant"].to(torch.float32) - parent_params[
            "mant"
        ].to(torch.float32)
        if not bool(torch.all((delta == 0.0) | (delta == 0.25) | (delta == -0.25))):
            raise AssertionError("joint arm produced a non +-0.25 mantissa step")
        if int((delta != 0.0).sum()) != changed_reported:
            raise AssertionError(
                "reported weight changed count disagrees with the mantissa delta"
            )
        if changed_reported <= 0 or not float(state.get("jrb1_w_delta_loss")) < 0.0:
            raise AssertionError("joint arm did not strictly decrease weight loss")
        if act_reported <= 0:
            raise AssertionError("joint arm reports no activation change")
        table = state.get(BOUNDARIES_KEY)
        if not torch.is_tensor(table) or int(table.numel()) != 7:
            raise AssertionError("joint arm did not store a length-7 table")
        values = table.to(torch.float32).reshape(-1) * 64.0
        if not bool(torch.all(values == torch.round(values))):
            raise AssertionError("activation table is not on the 1/64 grid")
        if not bool(torch.all((table >= 0.0) & (table <= 1.0))):
            raise AssertionError("activation table left the [0, 1] range")
        if int(state.get("jrb1_act_accepted", 0)) != 1:
            raise AssertionError("joint arm did not record activation acceptance")
        recount = 0
        for window in activations:
            base = mant_codes(
                parent.hif4_dynamic_quantize_activation(
                    window[0], window[1], state
                )
            )
            new = mant_codes(
                candidate.hif4_dynamic_quantize_activation(
                    window[0], window[1], state
                )
            )
            recount += int((new != base).sum())
        if recount != act_reported:
            raise AssertionError(
                f"activation recount {recount} != reported {act_reported}"
            )
        print(
            f"[D] joint arm: act_changed={act_reported} "
            f"w_changed={changed_reported} "
            f"w_delta_loss={float(state.get('jrb1_w_delta_loss')):.6e} "
            f"act_mse={float(state.get('jrb1_mse_parent')):.6e}->"
            f"{float(state.get('jrb1_mse_act_only')):.6e}"
        )
    else:
        if BOUNDARIES_KEY in state:
            raise AssertionError(f"arm {arm} still stored an activation table")
        if act_reported != 0:
            raise AssertionError(f"arm {arm} reports an activation change")
        for key, value in parent_params.items():
            if not bool(torch.equal(value, candidate_params[key])):
                raise AssertionError(f"arm {arm} changed weight field {key}")
        print(f"[D] {arm} arm: parent five fields restored bit for bit")

    deployed = candidate.hif4_dynamic_quantize_activation(
        activations[0][0], activations[0][1], state
    )
    assert_params(deployed, int(activations[0][0].shape[0]), channels)
    assert_params(candidate_params, rows, channels)
    return arm


def main() -> None:
    candidate_path = HERE / "candidate" / "solution.py"
    candidate = load_solution(candidate_path, "jrb1_candidate")
    parent = load_solution(ROOT / "solution.py", "jrb1_parent")
    reference = load_solution(
        ROOT / "evaluator" / "reference_hif4.py", "jrb1_reference"
    )
    single_file_import(candidate_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    generator = torch.Generator(device=device).manual_seed(227)
    # channels > _ACTIVATION_QUADRATIC_MAX_FEATURES (3072) is required: below
    # that cap the deployed activation mantissa comes from _adaround_mantissa
    # and the rounding boundary is structurally unreachable.
    rows, channels = 64, 3200
    weight_quant = torch.randn((rows, channels), generator=generator, device=device)
    weight_scale = torch.ones(
        (rows, channels // 16), dtype=torch.float32, device=device
    )
    activations = [
        (
            torch.randn((12, channels), generator=generator, device=device),
            torch.ones((12, channels // 16), dtype=torch.float32, device=device),
        ),
        (
            torch.randn((10, channels), generator=generator, device=device),
            torch.ones((10, channels // 16), dtype=torch.float32, device=device),
        ),
    ]

    parent_result = parent.hif4_calibration_and_quantize_weight(
        weight_quant, weight_scale, activations
    )
    result = candidate.hif4_calibration_and_quantize_weight(
        weight_quant, weight_scale, activations
    )
    if set(result) != {"weight_params", "activation_state"}:
        raise AssertionError(f"unexpected calibration keys: {set(result)}")
    assert_params(result["weight_params"], rows, channels)
    state = result["activation_state"]
    if state.get("jrb1_attempted") != 1:
        raise AssertionError("L-JRB1 did not reach its fixed boundary fit")
    if state.get("jrb1_coordinate") != (
        "joint-activation-weight-rounding-boundary"
    ):
        raise AssertionError("L-JRB1 coordinate provenance is missing")
    if state.get("jrb1_act_path") != "rounding-boundary":
        raise AssertionError(
            f"activation rounding path is not live: {state.get('jrb1_act_path')}"
        )
    if int(state.get("jrb1_boundary_count", 0)) != 18:
        raise AssertionError("L-JRB1 boundary count is not the fixed 18")
    if int(state.get("jrb1_fit_windows", 0)) != 2:
        raise AssertionError("L-JRB1 did not consume every supplied window")
    if state.get("jrb1_error"):
        raise AssertionError(f"L-JRB1 raised internally: {state['jrb1_error']}")

    control_b(candidate, device)
    control_c(candidate, parent, state, activations[0], device)
    arm = control_d(
        candidate,
        parent,
        result,
        parent_result,
        state,
        activations,
        rows,
        channels,
        device,
    )

    reference.validate_state(state)
    reference.validate_hif4_params(result["weight_params"], (rows, channels))
    print(
        f"L-JRB1 verify passed: device={device.type} arm={arm} "
        f"act_proposals={state.get('jrb1_act_proposals')} "
        f"w_proposals={state.get('jrb1_w_proposals')} "
        f"act_changed={state.get('jrb1_act_changed_mantissa')} "
        f"w_changed={state.get('jrb1_w_changed_mantissa')} "
        f"act_table={state.get('jrb1_act_boundaries').tolist()}"
    )


if __name__ == "__main__":
    main()
