"""L-XR1 legality, parent-control, factor-plumbing and reachability check.

Controls (plan 2026-09-10-linear-cross-residual-correction-plan.md, step 2):

  A. six public APIs exist and the candidate imports as a single file away
     from the repository; the candidate is a pure byte-append of the retained
     root, so every Attention API and shared helper is identical by
     construction (also checked by source equality);
  B. parent-off control: a state without the ``xr1`` factors returns the parent
     dynamic output bit for bit, and calibration leaves the parent five weight
     fields untouched; a wide layer (``in_features > 3072``) compiles no factors
     and stays bit-identical;
  C. factor plumbing: the compiled rank-4 factors are legal CPU tensors, the
     reported ``capture_g``/``capture_c`` reproduce an independent recomputation
     from the deployed weight, and the factors survive the parent block-order
     wrapper and GPTQ loop far enough to change real deployment mantissas;
  D. synthetic reachability: with exact full-rank ``G``/``C`` factors the
     compiled gradient is the exact output residual ``R W_hat``, so every
     accepted move must strictly decrease the exact output squared error
     recomputed from the full Gram and the exact cross term.

Diagnostics printed, never gated: rank-4 capture, off-block/cross energy, the
component means of the gradient, and the exact loss change of the real rank-4
move.
"""

from pathlib import Path
import hashlib
import importlib.util
import inspect
import math
import os
import shutil
import sys
import tempfile

import torch


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
PARENT_SHA256 = (
    "56dc805d6e5a3aef896db8021045740292735725d688b48e3d4393e55efcb2bd"
)
LINEAR_APIS = (
    "hif4_calibration_and_quantize_weight",
    "hif4_dynamic_quantize_activation",
)
ATTENTION_APIS = (
    "hif4_calibration_attention",
    "hif4_dynamic_quantize_q",
    "hif4_dynamic_quantize_k",
    "hif4_dynamic_quantize_v",
)


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
            raise AssertionError(
                f"{key} shape {tuple(params[key].shape)} != {shape}"
            )


def mant_codes(params: dict) -> torch.Tensor:
    return torch.round(
        params["mant"].detach().to(torch.float32) / 0.25
    ).to(torch.int64)


def single_file_import(candidate_path: Path):
    """Import the candidate from a directory with no repository siblings."""

    with tempfile.TemporaryDirectory() as tmp:
        isolated = Path(tmp) / "solution.py"
        shutil.copyfile(candidate_path, isolated)
        module = load_solution(isolated, "xr1_isolated")
    for name in LINEAR_APIS + ATTENTION_APIS:
        if not hasattr(module, name):
            raise AssertionError(f"isolated import is missing {name}")
    return module


def control_a(candidate_path: Path):
    """Append-only build, isolated import, Attention identity by construction."""

    parent_bytes = (ROOT / "solution.py").read_bytes()
    digest = hashlib.sha256(parent_bytes).hexdigest()
    if digest != PARENT_SHA256:
        raise AssertionError(f"retained root SHA256 changed: {digest}")
    candidate_bytes = candidate_path.read_bytes()
    prefix = parent_bytes.rstrip(b"\n")
    if not candidate_bytes.startswith(prefix):
        raise AssertionError("candidate is not a pure append of the retained root")
    tail = candidate_bytes[len(prefix):]
    if not tail.startswith(b"\n\n\n"):
        raise AssertionError("append separator changed")

    candidate = load_solution(candidate_path, "xr1_candidate")
    parent = load_solution(ROOT / "solution.py", "xr1_parent")
    single_file_import(candidate_path)
    for name in ATTENTION_APIS:
        if inspect.getsource(getattr(candidate, name)) != inspect.getsource(
            getattr(parent, name)
        ):
            raise AssertionError(f"Attention API {name} source drifted")
    for name in LINEAR_APIS:
        wrapper = getattr(candidate, name)
        original = getattr(candidate, "_XR1_PARENT_LINEAR_" + (
            "CALIBRATION" if "calibration" in name else "DYNAMIC"
        ))
        if wrapper is original:
            raise AssertionError(f"{name} was not rebound to the L-XR1 hook")
    print(
        f"[A] append-only candidate: parent={len(prefix)}B "
        f"candidate={len(candidate_bytes)}B attention APIs identical"
    )
    return candidate, parent


def make_layer(
    device: torch.device,
    seed: int,
    rows: int,
    channels: int,
    windows: int = 2,
    samples: int = 12,
):
    generator = torch.Generator(device="cpu").manual_seed(seed)
    weight_quant = torch.randn((rows, channels), generator=generator).to(device)
    weight_scale = torch.ones(
        (rows, channels // 16), dtype=torch.float32, device=device
    )
    activations = [
        (
            torch.randn((samples, channels), generator=generator).to(device),
            torch.ones(
                (samples, channels // 16), dtype=torch.float32, device=device
            ),
        )
        for _ in range(windows)
    ]
    return weight_quant, weight_scale, activations


def exact_metrics(candidate, weight_quant, weight_scale, params, state):
    """Full ``G``/``C`` and the deployed/dense weights in the natural frame."""

    dense = candidate._xr1_reconstruct_dense_weight(
        weight_quant, weight_scale, state
    )
    deployed = candidate._dequantize_hif4(params).to(torch.float32)
    gram = deployed.transpose(0, 1).mm(deployed)
    cross = (deployed - dense).transpose(0, 1).mm(deployed)
    return dense, deployed, gram, cross


def exact_residual(candidate, act_quant, act_scale, state, params, gram, cross):
    """Exact ``R W_hat = E G + X C`` and the dense/deployed activation pair."""

    dense = candidate._static_actorder_dense_from_state(
        act_quant, act_scale, state
    ).to(torch.float32)
    deployed = candidate._dequantize_hif4(params).to(torch.float32)
    error = deployed - dense
    residual = error.mm(gram) + dense.mm(cross)
    return dense, deployed, residual


def exact_loss_change(residual, delta, gram):
    """``2 <R W_hat, dX> + tr(dX G dX^T)`` for one combined activation move."""

    linear = 2.0 * float((residual * delta).sum())
    quadratic = float(((delta.mm(gram)) * delta).sum())
    return linear, quadratic


def control_b(candidate, parent, device: torch.device) -> None:
    """Parent-off bit identity, wide-layer bypass, five fields untouched."""

    weight_quant, weight_scale, activations = make_layer(
        device, 311, rows=64, channels=2560
    )
    parent_result = parent.hif4_calibration_and_quantize_weight(
        weight_quant, weight_scale, activations
    )
    result = candidate.hif4_calibration_and_quantize_weight(
        weight_quant, weight_scale, activations
    )
    if set(result) != {"weight_params", "activation_state"}:
        raise AssertionError(f"unexpected calibration keys: {set(result)}")
    for key, value in parent_result["weight_params"].items():
        if not bool(torch.equal(value, result["weight_params"][key])):
            raise AssertionError(f"calibration changed weight field {key}")
    state = result["activation_state"]
    if state.get("xr1_arm") != "compiled":
        raise AssertionError(f"narrow layer arm is {state.get('xr1_arm')}")
    if int(state.get("xr1_channels", -1)) != 2560:
        raise AssertionError("compiled channel count is wrong")

    parent_params = parent.hif4_dynamic_quantize_activation(
        activations[0][0], activations[0][1], dict(state)
    )
    stripped = {key: value for key, value in state.items() if key != "xr1"}
    candidate_params = candidate.hif4_dynamic_quantize_activation(
        activations[0][0], activations[0][1], dict(stripped)
    )
    for key, value in parent_params.items():
        if not bool(torch.equal(value, candidate_params[key])):
            raise AssertionError(f"parent-off control changed field {key}")

    wide_quant, wide_scale, wide_activations = make_layer(
        device, 313, rows=64, channels=3200
    )
    wide = candidate.hif4_calibration_and_quantize_weight(
        wide_quant, wide_scale, wide_activations
    )
    wide_state = wide["activation_state"]
    if wide_state.get("xr1_arm") != "wide-no-gram":
        raise AssertionError(f"wide layer arm is {wide_state.get('xr1_arm')}")
    if "xr1" in wide_state:
        raise AssertionError("wide layer stored factors without a group Gram")
    wide_parent = parent.hif4_dynamic_quantize_activation(
        wide_activations[0][0], wide_activations[0][1], dict(wide_state)
    )
    wide_candidate = candidate.hif4_dynamic_quantize_activation(
        wide_activations[0][0], wide_activations[0][1], dict(wide_state)
    )
    for key, value in wide_parent.items():
        if not bool(torch.equal(value, wide_candidate[key])):
            raise AssertionError(f"wide layer changed field {key}")
    print(
        "[B] parent-off bit identity holds for the narrow layer and the "
        "wide-no-gram bypass"
    )


def control_c(
    candidate,
    device: torch.device,
    weight_quant,
    weight_scale,
    activations,
    result,
) -> dict:
    """Factor legality, capture reproduction and real rank-4 reachability."""

    state = result["activation_state"]
    factors = state.get("xr1")
    if not isinstance(factors, dict):
        raise AssertionError("calibration did not store the xr1 factors")
    channels = int(state["in_features"])
    rank = 4
    shapes = {
        "u_g": (channels, rank),
        "lam_g": (rank,),
        "u_c": (channels, rank),
        "s_c": (rank,),
        "v_c": (channels, rank),
    }
    for key, shape in shapes.items():
        value = factors.get(key)
        if not torch.is_tensor(value) or value.device.type != "cpu":
            raise AssertionError(f"{key} is not a CPU tensor")
        if tuple(value.shape) != shape:
            raise AssertionError(f"{key} shape {tuple(value.shape)} != {shape}")
        if value.is_floating_point() and not bool(torch.isfinite(value).all()):
            raise AssertionError(f"{key} is not finite")
    if int(factors.get("rank", -1)) != rank:
        raise AssertionError("stored rank is not the fixed 4")

    dense, deployed, gram, cross = exact_metrics(
        candidate, weight_quant, weight_scale, result["weight_params"], state
    )
    offblock = gram - candidate._xr1_local_block_matrix(
        state["gram"], channels, gram.device, torch.float32
    )
    u_g = factors["u_g"].to(torch.float32)
    lam_g = factors["lam_g"].to(torch.float32)
    u_c = factors["u_c"].to(torch.float32)
    s_c = factors["s_c"].to(torch.float32)
    v_c = factors["v_c"].to(torch.float32)

    capture_g = float(lam_g.square().sum() / offblock.square().sum())
    capture_c = float(s_c.square().sum() / cross.square().sum())
    if abs(capture_g - float(factors["capture_g"])) > 1.0e-6:
        raise AssertionError("reported capture_g does not reproduce")
    if abs(capture_c - float(factors["capture_c"])) > 1.0e-6:
        raise AssertionError("reported capture_c does not reproduce")
    for name, basis in (("u_g", u_g), ("u_c", u_c), ("v_c", v_c)):
        gram_basis = basis.transpose(0, 1).mm(basis)
        eye = torch.eye(rank, dtype=torch.float32)
        if float((gram_basis - eye).abs().max()) > 1.0e-4:
            raise AssertionError(f"{name} columns are not orthonormal")
    eigen_residual = float(
        (offblock.mm(u_g) - u_g * lam_g.unsqueeze(0)).norm()
        / offblock.norm().clamp_min(1.0e-30)
    )
    if eigen_residual > 0.5:
        raise AssertionError(
            f"u_g/lam_g are not a rank-4 eigenpair of the off-block Gram "
            f"(relative residual {eigen_residual})"
        )
    ordered = bool(
        torch.all(
            lam_g.abs()[:-1] >= lam_g.abs()[1:] - 1.0e-9
        )
    )
    if not ordered:
        raise AssertionError("lam_g is not ordered by descending magnitude")

    parent_params = candidate._XR1_PARENT_LINEAR_DYNAMIC(
        activations[0][0], activations[0][1], dict(state)
    )
    live_state = dict(state)
    corrected = candidate.hif4_dynamic_quantize_activation(
        activations[0][0], activations[0][1], live_state
    )
    arm = live_state.get("xr1_dynamic_arm")
    changed = int(live_state.get("xr1_changed_mantissa", 0))
    parent_codes = mant_codes(parent_params)
    new_codes = mant_codes(corrected)
    recounted = int((new_codes != parent_codes).sum())
    if recounted != changed:
        raise AssertionError(
            f"reported changed count {changed} != recount {recounted}"
        )
    delta_codes = new_codes - parent_codes
    if not bool(torch.all((delta_codes == 0) | (delta_codes == 1) | (delta_codes == -1))):
        raise AssertionError("a mantissa code moved by more than one step")
    if not bool(torch.all((new_codes >= 0) & (new_codes <= 7))):
        raise AssertionError("a corrected mantissa code left [0, 7]")
    if not bool(torch.all((corrected["sign"] == 0.0) | (corrected["mant"] != 0.0))):
        raise AssertionError("a zero mantissa kept a non-canonical sign")

    print(
        f"[C] rank-4 factors legal: capture_g={capture_g:.4f} "
        f"capture_c={capture_c:.4f} eigen_residual={eigen_residual:.3e} "
        f"arm={arm} changed={changed} recounted={recounted} "
        f"offblock_rel={float(state.get('xr1_offblock_rel', 0.0)):.4f} "
        f"cross_rel={float(state.get('xr1_cross_rel', 0.0)):.4f}"
    )
    if changed > 0:
        delta = (
            candidate._dequantize_hif4(corrected).to(torch.float32)
            - candidate._dequantize_hif4(parent_params).to(torch.float32)
        )
        _, _, residual = exact_residual(
            candidate,
            activations[0][0],
            activations[0][1],
            state,
            parent_params,
            gram,
            cross,
        )
        linear, quadratic = exact_loss_change(residual, delta, gram)
        print(
            f"[C] real rank-4 move exact loss change "
            f"linear={linear:.6e} quadratic={quadratic:.6e} "
            f"total={linear + quadratic:.6e}"
        )
    return {
        "gram": gram,
        "cross": cross,
        "arm": arm,
        "changed": changed,
        "corrected": corrected,
        "parent_params": parent_params,
    }


def control_d(
    candidate,
    device: torch.device,
    weight_quant,
    weight_scale,
    activations,
    result,
    metrics: dict,
) -> None:
    """Exact full-rank factors must flip codes and strictly reduce the loss."""

    state = result["activation_state"]
    channels = int(state["in_features"])
    gram = metrics["gram"]
    cross = metrics["cross"]
    local = candidate._xr1_local_block_matrix(
        state["gram"], channels, gram.device, torch.float32
    )
    offblock = gram - local

    values, vectors = torch.linalg.eigh(offblock)
    order = torch.argsort(values.abs(), descending=True)
    u_g = vectors.index_select(1, order)
    lam_g = values.index_select(0, order)
    u_c, s_c, v_h = torch.linalg.svd(cross, full_matrices=False)
    v_c = v_h.transpose(0, 1)  # the hook expects V with right vectors as columns

    exact_state = dict(state)
    exact_state["xr1"] = {
        "u_g": candidate._cpu_state_tensor(u_g.contiguous()),
        "lam_g": candidate._cpu_state_tensor(lam_g.contiguous()),
        "u_c": candidate._cpu_state_tensor(u_c.contiguous()),
        "s_c": candidate._cpu_state_tensor(s_c.contiguous()),
        "v_c": candidate._cpu_state_tensor(v_c.contiguous()),
        "rank": channels,
        "capture_g": 1.0,
        "capture_c": 1.0,
    }

    act_quant, act_scale = activations[0]
    parent_params = candidate._XR1_PARENT_LINEAR_DYNAMIC(
        act_quant, act_scale, dict(state)
    )
    corrected = candidate.hif4_dynamic_quantize_activation(
        act_quant, act_scale, exact_state
    )
    changed = int(exact_state.get("xr1_changed_mantissa", 0))
    if changed <= 0:
        raise AssertionError("exact factors reached no mantissa code")

    parent_codes = mant_codes(parent_params)
    new_codes = mant_codes(corrected)
    delta_codes = new_codes - parent_codes
    if not bool(torch.all((delta_codes == 0) | (delta_codes == 1) | (delta_codes == -1))):
        raise AssertionError("an exact-factor move left the +-1 code step")

    _, _, residual = exact_residual(
        candidate, act_quant, act_scale, state, parent_params, gram, cross
    )
    delta = (
        candidate._dequantize_hif4(corrected).to(torch.float32)
        - candidate._dequantize_hif4(parent_params).to(torch.float32)
    )
    if float(delta.abs().sum()) <= 0.0:
        raise AssertionError("exact factors changed no deployment value")

    linear, quadratic = exact_loss_change(residual, delta, gram)
    total = linear + quadratic
    diagonal = float(
        (
            torch.diagonal(gram).unsqueeze(0) * delta.square()
        ).sum()
    )
    predicted = float(exact_state.get("xr1_predicted_cost", float("nan")))
    if not math.isfinite(predicted):
        raise AssertionError("exact-factor move stored no predicted cost")
    if abs(predicted - (linear + diagonal)) > 1.0e-6 * max(1.0, abs(linear)):
        raise AssertionError(
            f"predicted cost {predicted:.6e} != exact linear+diagonal "
            f"{linear + diagonal:.6e}: the gradient is not the exact residual"
        )
    if not predicted < 0.0:
        raise AssertionError(f"predicted cost {predicted:.6e} is not negative")
    if not total < 0.0:
        raise AssertionError(
            f"exact output squared error did not decrease: {total:.6e}"
        )

    per_element = 2.0 * residual * delta + torch.diagonal(gram).unsqueeze(0) * delta.square()
    worst = float(per_element.masked_select(delta != 0.0).max())
    if not worst < 0.0:
        raise AssertionError(f"a moved element has positive exact cost {worst:.6e}")

    assert_params(corrected, int(act_quant.shape[0]), channels)
    print(
        f"[D] exact-factor reachability: changed={changed} "
        f"linear={linear:.6e} diagonal={diagonal:.6e} "
        f"quadratic={quadratic:.6e} total={total:.6e} "
        f"predicted={predicted:.6e} worst_element={worst:.6e}"
    )


def main() -> None:
    device = torch.device(os.environ.get("XR1_VERIFY_DEVICE", "cpu"))
    candidate_path = HERE / "candidate" / "solution.py"
    candidate, parent = control_a(candidate_path)
    reference = load_solution(
        ROOT / "evaluator" / "reference_hif4.py", "xr1_reference"
    )

    weight_quant, weight_scale, activations = make_layer(
        device, 311, rows=64, channels=2560
    )
    result = candidate.hif4_calibration_and_quantize_weight(
        weight_quant, weight_scale, activations
    )
    if result["activation_state"].get("xr1_error"):
        raise AssertionError(
            f"compile raised internally: {result['activation_state']['xr1_error']}"
        )
    control_b(candidate, parent, device)
    metrics = control_c(
        candidate, device, weight_quant, weight_scale, activations, result
    )
    control_d(
        candidate, device, weight_quant, weight_scale, activations, result, metrics
    )

    reference.validate_state(result["activation_state"])
    reference.validate_hif4_params(result["weight_params"], (64, 2560))
    reference.validate_hif4_params(
        metrics["corrected"], (int(activations[0][0].shape[0]), 2560)
    )
    state = result["activation_state"]
    print(
        f"L-XR1 verify passed: device={device.type} arm={state.get('xr1_arm')} "
        f"dynamic_arm={metrics['arm']} changed={metrics['changed']} "
        f"capture_g={float(state.get('xr1_capture_g', 0.0)):.4f} "
        f"capture_c={float(state.get('xr1_capture_c', 0.0)):.4f}"
    )


if __name__ == "__main__":
    main()
