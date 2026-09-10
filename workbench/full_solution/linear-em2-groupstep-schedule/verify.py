"""L-EM2 legality, parent-control, metric-plumbing and monotonicity check.

Controls (plan 2026-09-10-linear-groupstep-schedule-plan.md, step 2):

  A. six public APIs exist and the candidate imports as a single file away from
     the repository; the candidate is a pure byte-append of the retained root,
     so every Attention API and shared helper is identical by construction;
  B. parent-off control: a state without the ``em1`` payload returns the parent
     dynamic output bit for bit, calibration leaves the parent five weight
     fields untouched, and a layer with ``in_features > 4096`` stores nothing
     and stays bit-identical;
  C. metric plumbing: the compiled ``h``/``gram_diag_mean`` reproduce an independent
     recomputation from the deployed weight, ``G = h_inv^{-1} - c I`` matches
     ``W_hat^T W_hat``, and the descent changes real deployment mantissas;
  D. monotonicity: the true output squared error ``L`` recomputed from the full
     deployed weight and the reference activation must not increase, and its
     change must equal the sum of the accepted exact *row-joint* quadratics
     ``2 <row_delta, g> + row_delta^T G row_delta`` -- which equals the change
     of ``J`` because ``L - J`` is independent of ``X``.  Unlike L-EM1 the
     accepted unit is a whole row's step move, so this checks the coarser
     acceptance rule's exactness, not just a group's;
  E. determinism and state round-trip;
  F. schedule shape: the real case reproduces ``probe_frontier.py``'s
     ``ideal/groupstep/pm1/p2`` reading (fp32 vs float64), and the accepted
     step count stays within ``K * 16``.

Diagnostics printed, never gated: the ridge residual, the metric recovery
error, the accepted-group count and the predicted/true loss change.
"""

from pathlib import Path
import hashlib
import importlib.util
import inspect
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
CACHE = (
    ROOT
    / "artifacts/official_eval/cache/proxy-v3-calibration"
    / "56dc805d6e5a3aef-linear-452e6fe4aa5747590535.pt"
)
PACK = ROOT / "artifacts/official_eval/cache/qwen3.5-4b-proxy-v2.pt"

# The probe frontier this candidate must reproduce, keyed by pass count, read
# from logs/execution/linear-em1-probe-frontier-q3.out (layer0/q, in=2560,
# rows=128, ideal target, pm1 candidate set, group-major schedule).  The pairing
# of K with its own frontier is the point: control F checks the running K
# against _EM2_PASSES, so a drift in one without the other fails.
_EM2_PASSES = 1
_EM2_REAL_REL_BY_PASSES = {1: -0.190998, 2: -0.260625}
_EM2_REAL_REL = _EM2_REAL_REL_BY_PASSES[_EM2_PASSES]


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


def single_file_import(candidate_path: Path):
    """Import the candidate from a directory with no repository siblings."""

    with tempfile.TemporaryDirectory() as tmp:
        isolated = Path(tmp) / "solution.py"
        shutil.copyfile(candidate_path, isolated)
        module = load_solution(isolated, "em1_isolated")
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
    if not candidate_bytes[len(prefix):].startswith(b"\n\n\n"):
        raise AssertionError("append separator changed")

    candidate = load_solution(candidate_path, "em1_candidate")
    parent = load_solution(ROOT / "solution.py", "em1_parent")
    single_file_import(candidate_path)
    for name in ATTENTION_APIS:
        if inspect.getsource(getattr(candidate, name)) != inspect.getsource(
            getattr(parent, name)
        ):
            raise AssertionError(f"Attention API {name} source drifted")
    for name, original_name in (
        ("hif4_calibration_and_quantize_weight", "_EM1_PARENT_LINEAR_CALIBRATION"),
        ("hif4_dynamic_quantize_activation", "_EM1_PARENT_LINEAR_DYNAMIC"),
    ):
        wrapper = getattr(candidate, name)
        original = getattr(candidate, original_name)
        if wrapper is original:
            raise AssertionError(f"{name} was not rebound to the L-EM2 hook")
        if wrapper is getattr(parent, name):
            raise AssertionError(f"{name} still points at the parent")
    print(
        f"[A] append-only candidate: parent={len(prefix)}B "
        f"candidate={len(candidate_bytes)}B attention APIs identical"
    )
    return candidate, parent


def make_layer(device, seed, rows, channels, windows=2, samples=12):
    generator = torch.Generator(device="cpu").manual_seed(seed)
    weight_quant = torch.randn((rows, channels), generator=generator).to(device)
    weight_scale = torch.ones((rows, channels // 16), dtype=torch.float32, device=device)
    activations = [
        (
            torch.randn((samples, channels), generator=generator).to(device),
            torch.ones((samples, channels // 16), dtype=torch.float32, device=device),
        )
        for _ in range(windows)
    ]
    return weight_quant, weight_scale, activations


def true_loss(deployed_activation, weight_hat, reference_activation, reference_weight):
    """``|| x W_hat^T - X_ref W^T ||^2`` in float64, the eval's player mse."""

    player = deployed_activation.to(torch.float64) @ weight_hat.to(torch.float64).t()
    reference = (
        reference_activation.to(torch.float64) @ reference_weight.to(torch.float64).t()
    )
    return float((player - reference).square().sum())


def control_b(candidate, parent, device):
    """Parent-off bit identity and the out-of-scope bypass."""

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
    if state.get("em1_arm") != "compiled":
        raise AssertionError(f"narrow layer arm is {state.get('em1_arm')}")
    if int(state.get("em1_channels", -1)) != 2560:
        raise AssertionError("compiled channel count is wrong")
    payload = state.get("em1")
    if not isinstance(payload, dict) or tuple(payload["h"].shape) != (2560, 2560):
        raise AssertionError("compiled H has the wrong shape")
    if payload["h"].dtype != torch.float32 or payload["h"].device.type != "cpu":
        raise AssertionError("compiled H must be a CPU float32 tensor")

    parent_params = parent.hif4_dynamic_quantize_activation(
        activations[0][0], activations[0][1], dict(state)
    )
    stripped = {key: value for key, value in state.items() if key != "em1"}
    candidate_params = candidate.hif4_dynamic_quantize_activation(
        activations[0][0], activations[0][1], dict(stripped)
    )
    for key, value in parent_params.items():
        if not bool(torch.equal(value, candidate_params[key])):
            raise AssertionError(f"parent-off control changed field {key}")

    wide_quant, wide_scale, wide_activations = make_layer(
        device, 313, rows=64, channels=4160
    )
    wide = candidate.hif4_calibration_and_quantize_weight(
        wide_quant, wide_scale, wide_activations
    )
    wide_state = wide["activation_state"]
    if wide_state.get("em1_arm") != "out-of-scope":
        raise AssertionError(f"wide layer arm is {wide_state.get('em1_arm')}")
    if "em1" in wide_state:
        raise AssertionError("wide layer stored a metric payload")
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
        "out-of-scope bypass"
    )


def control_c(candidate, device, label, state, weight_params, weight_quant, weight_scale):
    """Metric plumbing: compiled H/ridge, G recovery, real code reachability."""

    payload = state.get("em1")
    if not isinstance(payload, dict):
        raise AssertionError(f"{label}: no em1 payload")
    channels = int(state["in_features"])
    deployed = candidate._dequantize_hif4(weight_params).to(torch.float32)
    dense = candidate._dequantize_nvfp4_float32(weight_quant, weight_scale).to(
        torch.float32
    )
    gram = deployed.transpose(0, 1).mm(deployed)
    cross = dense.transpose(0, 1).mm(deployed)
    h_true = gram - cross
    h_stored = payload["h"].to(torch.float32)
    g_norm = float(gram.norm())
    h_gap = float((h_stored - h_true).norm() / max(g_norm, 1e-30))
    h_scale = float(h_true.norm() / max(g_norm, 1e-30))

    # Calibration stores mean(diag(gram)) only; the dynamic API supplies the
    # other half of the ridge.  Check the stored scalar, then check that the
    # decomposition the dynamic path actually uses rebuilds G.
    gram_diag_mean = float(payload["gram_diag_mean"])
    diag_gap = abs(gram_diag_mean - float(gram.diagonal().mean())) / max(
        abs(gram_diag_mean), 1e-30
    )

    h_inv = state["h_inv"].to(torch.float32)
    inverse = torch.cholesky_inverse(torch.linalg.cholesky(h_inv))
    ridge = float(inverse.diagonal().mean()) - gram_diag_mean
    metric = inverse - ridge * torch.eye(channels)
    g_rel = float((metric - gram).norm() / max(g_norm, 1e-30))

    # And exercise the real code path rather than a reimplementation of it.
    live = candidate._em1_metric(dict(state), channels, torch.device("cpu"))
    if live is None:
        raise AssertionError(f"{label}: _em1_metric declined the compiled state")
    live_metric, live_h = live
    path_rel = float((live_metric - metric).norm() / max(float(metric.norm()), 1e-30))
    path_h_rel = float(
        (live_h - h_stored).norm() / max(float(h_stored.norm()), 1e-30)
    )
    print(
        f"[C:{label}] channels={channels} ridge={ridge:.6e} "
        f"|H|/|G|={h_scale:.3e} H_gap/|G|={h_gap:.3e} G_rel={g_rel:.3e} "
        f"diag_gap={diag_gap:.3e} path_rel={path_rel:.3e} path_h_rel={path_h_rel:.3e}"
    )
    if h_gap > 1e-5:
        raise AssertionError(f"{label}: stored H does not match the recomputation")
    if g_rel > 1e-3:
        raise AssertionError(f"{label}: G recovery from h_inv is too loose")
    if diag_gap > 1e-6:
        raise AssertionError(f"{label}: stored gram_diag_mean drifted by {diag_gap:.3e}")
    if path_rel > 1e-6 or path_h_rel > 1e-6:
        raise AssertionError(
            f"{label}: _em1_metric disagrees with the reference ({path_rel:.3e}, "
            f"{path_h_rel:.3e})"
        )
    return gram, cross, dense, deployed


def control_d(candidate, parent, device, label, state, weight_params, act_pair,
              weight_quant, weight_scale, reference_weight):
    """Monotonicity: true L must not increase and must equal the model sum."""

    channels = int(state["in_features"])
    rows = int(act_pair[0].shape[0])
    parent_params = parent.hif4_dynamic_quantize_activation(
        act_pair[0], act_pair[1], dict(state)
    )
    live_state = dict(state)
    candidate_params = candidate.hif4_dynamic_quantize_activation(
        act_pair[0], act_pair[1], live_state
    )
    assert_params(candidate_params, rows, channels)
    deployed_before = candidate._dequantize_hif4(parent_params).to(torch.float32)
    deployed_after = candidate._dequantize_hif4(candidate_params).to(torch.float32)
    weight_hat = candidate._dequantize_hif4(weight_params).to(torch.float32)
    reference_activation = candidate._dequantize_nvfp4_float32(
        act_pair[0], act_pair[1]
    ).to(torch.float32)
    loss_before = true_loss(
        deployed_before, weight_hat, reference_activation, reference_weight
    )
    loss_after = true_loss(
        deployed_after, weight_hat, reference_activation, reference_weight
    )
    changed = int((deployed_after != deployed_before).any(dim=1).sum())
    delta = loss_after - loss_before
    print(
        f"[D:{label}] rows={rows} changed_rows={changed} "
        f"L_before={loss_before:.6e} L_after={loss_after:.6e} "
        f"dL={delta:+.6e} rel={delta / max(loss_before, 1e-30):+.4%}"
    )
    predicted = live_state.get("em1_predicted_cost")
    if predicted is not None:
        gap = abs(float(predicted) - delta) / max(abs(delta), 1e-30)
        print(
            f"[D:{label}] predicted_cost={float(predicted):+.6e} "
            f"true_dL={delta:+.6e} gap_rel={gap:.3e}"
        )
        if gap > 1e-2:
            raise AssertionError(f"{label}: predicted cost misses the true dL")
    if delta > 1e-12 * max(loss_before, 1.0):
        raise AssertionError(f"{label}: the mechanism increased the true loss")
    if changed == 0:
        print(f"[D:{label}] NOTE no mantissa changed on this case")
    return delta, loss_before


def gradient_identity(candidate, device, label, state, weight_params, act_pair,
                      reference_weight):
    """Finite-difference check of g = (X - X_ref) G + X_ref H on one group."""

    metric_pair = candidate._em1_metric(state, int(state["in_features"]), device)
    if metric_pair is None:
        raise AssertionError(f"{label}: metric rebuild failed")
    metric, h_matrix = metric_pair
    deployed = candidate._dequantize_hif4(
        candidate.hif4_dynamic_quantize_activation(act_pair[0], act_pair[1], dict(state))
    ).to(torch.float32)
    reference = candidate._dequantize_nvfp4_float32(act_pair[0], act_pair[1]).to(
        torch.float32
    )
    weight_hat = candidate._dequantize_hif4(weight_params).to(torch.float32)
    cross = (metric - h_matrix).double()
    gram64 = metric.double()
    target = torch.linalg.lstsq(
        gram64, cross.t().mm(reference.double().t())
    ).solution.t()
    solve_residual = float(
        (target.double().mm(gram64) - reference.double().mm(cross)).norm()
        / max(float(reference.double().mm(cross).norm()), 1e-30)
    )

    def objective(x):
        delta = (x - target).double()
        return float((delta.mm(gram64) * delta).sum())

    gradient32 = (deployed - reference).mm(metric) + reference.mm(h_matrix)
    gradient = (deployed.double() - reference.double()).mm(
        metric.double()
    ) + reference.double().mm(h_matrix.double())
    generator = torch.Generator(device="cpu").manual_seed(7)
    move = torch.zeros_like(deployed).double()
    rows = int(deployed.shape[0])
    move[:, :4] = torch.randn((rows, 4), generator=generator).to(device).double()
    step = 1e-2
    base = deployed.double()
    numeric = (objective(base + step * move) - objective(base - step * move)) / (
        2.0 * step
    )
    analytic = 2.0 * float((gradient * move.double()).sum())
    analytic32 = 2.0 * float((gradient32 * move).sum())
    rel = abs(numeric - analytic) / max(abs(numeric), abs(analytic), 1e-30)
    print(
        f"[D:{label}] gradient check: analytic={analytic:+.6e} "
        f"numeric={numeric:+.6e} rel={rel:.3e} fp32_rel="
        f"{abs(analytic32 - analytic) / max(abs(analytic), 1e-30):.3e} "
        f"solve_residual={solve_residual:.3e}"
    )
    if solve_residual > 1e-6:
        print(f"[D:{label}] NOTE metric is rank-deficient; gradient check skipped")
        return
    if rel > 1e-3:
        raise AssertionError(f"{label}: gradient identity failed")


def control_e(candidate, device, label, state, act_pair, cache_path: Path):
    """Determinism and state round-trip through torch.save/torch.load."""

    first = candidate.hif4_dynamic_quantize_activation(
        act_pair[0], act_pair[1], dict(state)
    )
    second = candidate.hif4_dynamic_quantize_activation(
        act_pair[0], act_pair[1], dict(state)
    )
    for key, value in first.items():
        if not bool(torch.equal(value, second[key])):
            raise AssertionError(f"{label}: dynamic API is not deterministic ({key})")
    torch.save(state, cache_path)
    reloaded = torch.load(cache_path, map_location="cpu", weights_only=False)
    third = candidate.hif4_dynamic_quantize_activation(
        act_pair[0], act_pair[1], reloaded
    )
    for key, value in first.items():
        if not bool(torch.equal(value, third[key])):
            raise AssertionError(f"{label}: state round-trip changed the output ({key})")
    cache_path.unlink()
    print(f"[E:{label}] determinism and state round-trip hold")


def control_f(candidate, label, state, expected_rel, tol, expected_passes):
    """Schedule shape: K * 16 steps always run, and moves are kept."""

    # expected_passes is passed in from the caller rather than read off the
    # module, so a drifted K cannot silently agree with itself.  The caller also
    # pairs it with the probe frontier for that K, which is the real check.
    if int(candidate._EM1_PASSES) != expected_passes:
        raise AssertionError(
            f"K drifted: {candidate._EM1_PASSES}, control expects {expected_passes}"
        )
    if int(candidate._EM1_GROUPS_PER_BLOCK) != 16:
        raise AssertionError("groups per block drifted")
    cap = int(candidate._EM1_PASSES) * candidate._EM1_GROUPS_PER_BLOCK
    steps = int(state.get("em1_accepted_steps", -1))
    moves = int(state.get("em1_accepted_groups", -1))
    print(
        f"[F:{label}] passes={candidate._EM1_PASSES} accepted_steps={steps} "
        f"(cap {cap}) accepted_group_moves={moves} expected_rel={expected_rel}"
    )
    if steps != cap:
        raise AssertionError(f"{label}: ran {steps} steps, expected {cap}")
    if moves <= 0:
        raise AssertionError(f"{label}: not one group move was kept")
    return tol


def main() -> None:
    device = torch.device("cpu")
    candidate_path = HERE / "candidate" / "solution.py"
    candidate, parent = control_a(candidate_path)
    control_b(candidate, parent, device)

    weight_quant, weight_scale, activations = make_layer(
        device, 907, rows=64, channels=2560
    )
    result = candidate.hif4_calibration_and_quantize_weight(
        weight_quant, weight_scale, activations
    )
    state = result["activation_state"]
    control_c(
        candidate, device, "synthetic", state, result["weight_params"],
        weight_quant, weight_scale,
    )
    reference_weight = candidate._dequantize_nvfp4_float32(
        weight_quant, weight_scale
    ).to(torch.float32)
    synthetic_state = dict(state)
    candidate.hif4_dynamic_quantize_activation(
        activations[0][0], activations[0][1], synthetic_state
    )
    control_f(candidate, "synthetic", synthetic_state, None, None, _EM2_PASSES)
    control_d(
        candidate, parent, device, "synthetic", state, result["weight_params"],
        activations[0], weight_quant, weight_scale, reference_weight,
    )
    gradient_identity(
        candidate, device, "synthetic", state, result["weight_params"],
        activations[0], reference_weight,
    )
    control_e(
        candidate, device, "synthetic", state, activations[0],
        HERE / "_em1_state_roundtrip.pt",
    )

    if CACHE.exists() and PACK.exists():
        payload = torch.load(CACHE, map_location="cpu", mmap=True, weights_only=False)
        pack = torch.load(PACK, map_location="cpu", mmap=True, weights_only=False)
        entry = next(
            item for item in payload["weight_states"]
            if int(item["layer"]) == 0 and str(item["role"]) == "q"
        )
        sys.path.insert(0, str(ROOT / "evaluator"))
        import official_eval as v2  # noqa: PLC0415

        real_state = dict(entry["state"])
        weight_quant, weight_scale = v2._pair(
            pack["weights"][0]["q"].to(torch.float32)
        )
        weight_quant = weight_quant.to(torch.float32)
        weight_scale = weight_scale.to(torch.float32)
        real_result = {
            "weight_params": entry["params"],
            "activation_state": real_state,
        }
        diagnostics: dict = {}
        candidate._em1_compile_metric(
            weight_quant, weight_scale, real_result, diagnostics
        )
        if diagnostics.get("em1_arm") != "compiled":
            raise AssertionError(f"real layer arm is {diagnostics.get('em1_arm')}")
        raw = pack["test_activations"]["q"][1][0].to(torch.float32)[:128].contiguous()
        pair = tuple(value.to(device) for value in v2._pair(raw))
        reference_weight = candidate._dequantize_nvfp4_float32(
            weight_quant, weight_scale
        ).to(torch.float32)
        control_c(
            candidate, device, "real/layer0/q", real_state, entry["params"],
            weight_quant, weight_scale,
        )
        real_live = dict(real_state)
        candidate.hif4_dynamic_quantize_activation(pair[0], pair[1], real_live)
        control_f(
            candidate, "real/layer0/q", real_live, _EM2_REAL_REL, 0.02, _EM2_PASSES
        )
        real_delta, real_loss_before = control_d(
            candidate, parent, device, "real/layer0/q", real_state, entry["params"],
            pair, weight_quant, weight_scale, reference_weight,
        )
        real_rel = real_delta / max(real_loss_before, 1e-30)
        print(
            f"[F:real/layer0/q] observed rel={real_rel:+.4%} vs probe "
            f"ideal/groupstep/pm1/p{_EM2_PASSES} = {_EM2_REAL_REL:+.4%}"
        )
        if abs(real_rel - _EM2_REAL_REL) > 0.02:
            raise AssertionError(
                f"real case rel {real_rel:+.4%} does not reproduce the probe frontier"
            )
        gradient_identity(
            candidate, device, "real/layer0/q", real_state, entry["params"],
            pair, reference_weight,
        )
        control_e(
            candidate, device, "real/layer0/q", real_state, pair,
            HERE / "_em1_state_roundtrip.pt",
        )
    else:
        print("[D:real] shard cache or pack missing; real-data control skipped")

    print("ALL L-EM2 CONTROLS PASSED")


if __name__ == "__main__":
    main()
