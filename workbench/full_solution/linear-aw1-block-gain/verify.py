"""Verify v197 linear-aw1-block-gain on small synthetic Linear cases.

Checks: closed-form sufficient-statistics solution executes and matches an
explicit per-fold Z recomputation, gains are non-identity on improving data
(reachable), the hard gate accepts on improving data and rejects on
degenerate (zero) data, the force-identity control path is bit-identical to
the root parent, legal five-field state via evaluator/reference_hif4.py,
finite outputs through the deployed dynamic API, a realistic-size timing
probe, and a repo-free single-file six-API import check in a subprocess.
"""

from pathlib import Path
import hashlib
import importlib.util
import json
import math
import shutil
import subprocess
import sys
import tempfile
import time

import torch


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "evaluator"))
import reference_hif4 as ref

torch.set_num_threads(1)


def load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_linear_case(
    device: torch.device,
    *,
    out_features: int = 256,
    in_features: int = 256,
    windows: int = 6,
    tokens: int = 128,
    seed: int = 197,
    spike: float = 0.0,
    zero: bool = False,
):
    generator = torch.Generator(device="cpu").manual_seed(seed)
    weight = torch.randn(out_features, in_features, generator=generator) * 0.1
    w_scale = torch.ones(out_features, in_features // 16)
    calib = []
    for index in range(windows):
        a = torch.randn(tokens, in_features, generator=generator)
        if spike > 0.0:
            # One dominant channel per 64-block: block-wise activation
            # quantization then shrinks the remaining channels, so a per-block
            # gain genuinely reduces the deployed output error and the gate
            # must accept.
            for block in range(in_features // 64):
                a[:, block * 64 + 13] *= spike
        if zero:
            a = torch.zeros_like(a)
        a_scale = torch.ones(tokens, in_features // 16)
        calib.append((a, a_scale))
    return (
        weight.to(device),
        w_scale.to(device),
        [(a.to(device), s.to(device)) for a, s in calib],
    )


def explicit_quad_losses(candidate, weight_quant, weight_scale, calib, state, parent_params, gains):
    """Recompute L(g) and L(1) with explicit per-fold Z (no statistics)."""

    weight = candidate._dequantize_nvfp4_float32(weight_quant, weight_scale).to(
        torch.float32
    )
    w_hat = candidate._dequantize_hif4(parent_params).to(torch.float32)
    w_std = candidate._dequantize_hif4(
        candidate._branch_encode_standard_hif4(weight)
    ).to(torch.float32)
    blocks = int(weight.shape[1]) // 64
    loss_one = 0.0
    loss_gain = 0.0
    for pair in calib[: candidate._AW1_MAX_WINDOWS]:
        a_q = candidate._sample_rows(pair[0], candidate._AW1_MAX_TOKENS)
        a_s = candidate._sample_rows(pair[1], candidate._AW1_MAX_TOKENS)
        a_fp = candidate._dequantize_nvfp4_float32(a_q, a_s).to(torch.float32)
        a_hat = candidate._dequantize_hif4(
            candidate.hif4_dynamic_quantize_activation(a_q, a_s, state)
        ).to(torch.float32)
        a_std = candidate._dequantize_hif4(
            candidate._branch_encode_standard_hif4(a_fp)
        ).to(torch.float32)
        y_ref = a_fp @ weight.t()
        y_std = a_std @ w_std.t()
        omega = 1.0 / (float((y_ref - y_std).square().mean()) + candidate._EPS)
        z = torch.stack(
            [
                a_hat[:, b * 64 : (b + 1) * 64]
                @ w_hat[:, b * 64 : (b + 1) * 64].t()
                for b in range(blocks)
            ]
        )  # [B, T, out]
        g = gains.to(z.device).reshape(blocks, 1, 1)
        loss_gain += omega * float((y_ref - (g * z).sum(dim=0)).square().sum())
        loss_one += omega * float((y_ref - z.sum(dim=0)).square().sum())
    return loss_one, loss_gain


def main():
    candidate = load(HERE / "candidate" / "solution.py", "aw1_verify_candidate")
    parent = load(ROOT / "solution.py", "aw1_verify_parent")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    result = {
        "candidate_sha256": hashlib.sha256(
            (HERE / "candidate" / "solution.py").read_bytes()
        ).hexdigest(),
        "parent_sha256": hashlib.sha256((ROOT / "solution.py").read_bytes()).hexdigest(),
        "device": device.type,
    }

    # --- Acceptance case: wide plain-codec layer has real block-gain room ---
    weight, w_scale, calib = make_linear_case(
        device, out_features=512, in_features=4096, windows=6, tokens=128
    )
    with torch.inference_mode():
        accepted = candidate.hif4_calibration_and_quantize_weight(
            weight, w_scale, calib
        )
        parent_result = parent.hif4_calibration_and_quantize_weight(
            weight, w_scale, calib
        )
    state = accepted["activation_state"]
    assert state["aw1_attempted"] == 1, state
    assert state["aw1_accepted"] == 1, state.get("aw1_arm")
    assert state["aw1_arm"] == "accepted"
    assert state["aw1_fit_windows"] == 1
    assert state["aw1_loss_candidate"] < state["aw1_loss_parent"]
    gains = state["aw1_gains"].to(torch.float64)
    gain_dev = float((gains - 1.0).abs().mean())
    assert gain_dev > 0.0
    result["acceptance_case"] = "PASS"
    result["gain_abs_dev_mean"] = gain_dev
    result["loss_parent"] = state["aw1_loss_parent"]
    result["loss_candidate"] = state["aw1_loss_candidate"]
    result["closed_form_relative_improvement"] = (
        1.0 - state["aw1_loss_candidate"] / state["aw1_loss_parent"]
    )

    # Sufficient-statistics cross-check: the gate numbers must match an
    # explicit per-fold Z evaluation of the same quadratic objective.
    exp_one, exp_gain = explicit_quad_losses(
        candidate, weight, w_scale, calib, state,
        parent_result["weight_params"], gains,
    )
    rel_one = abs(exp_one - state["aw1_loss_parent"]) / max(abs(exp_one), 1e-30)
    rel_gain = abs(exp_gain - state["aw1_loss_candidate"]) / max(abs(exp_gain), 1e-30)
    assert rel_one < 1e-4, (exp_one, state["aw1_loss_parent"])
    assert rel_gain < 1e-4, (exp_gain, state["aw1_loss_candidate"])
    result["sufficient_statistics_match"] = "PASS"
    result["suffstats_rel_err"] = [rel_one, rel_gain]

    # Legal state and finite deployed outputs.
    ref.validate_state(state)
    out_shape = (int(weight.shape[0]), int(weight.shape[1]))
    ref.validate_hif4_params(accepted["weight_params"], out_shape)
    assert bool(
        torch.isfinite(candidate._dequantize_hif4(accepted["weight_params"])).all()
    )
    act_params = candidate.hif4_dynamic_quantize_activation(
        calib[0][0], calib[0][1], state
    )
    ref.validate_hif4_params(act_params, calib[0][0].shape)
    assert bool(torch.isfinite(candidate._dequantize_hif4(act_params)).all())
    result["state_and_output_contract"] = "PASS"

    # Deployed weight differs from the parent only where g != 1.
    delta = (
        candidate._dequantize_hif4(accepted["weight_params"]).to(torch.float32)
        - candidate._dequantize_hif4(parent_result["weight_params"]).to(torch.float32)
    )
    assert float(delta.abs().max()) > 0.0
    result["deployed_weight_changed"] = "PASS"

    # --- Rejection case: zero activations give a degenerate objective -------
    z_weight, z_wscale, z_calib = make_linear_case(device, zero=True)
    with torch.inference_mode():
        rejected = candidate.hif4_calibration_and_quantize_weight(
            z_weight, z_wscale, z_calib
        )
        z_parent = parent.hif4_calibration_and_quantize_weight(
            z_weight, z_wscale, z_calib
        )
    z_state = rejected["activation_state"]
    assert z_state["aw1_attempted"] == 1
    assert z_state["aw1_accepted"] == 0
    assert z_state["aw1_arm"] == "parent"
    for key, value in z_parent["weight_params"].items():
        assert torch.equal(rejected["weight_params"][key], value)
    ref.validate_state(z_state)
    result["degenerate_rejection"] = "PASS"

    # --- Control: force-identity gains must be bit-identical to the parent --
    candidate._AW1_FORCE_IDENTITY = True
    try:
        with torch.inference_mode():
            control = candidate.hif4_calibration_and_quantize_weight(
                weight, w_scale, calib
            )
    finally:
        candidate._AW1_FORCE_IDENTITY = False
    c_state = control["activation_state"]
    assert c_state["aw1_attempted"] == 1
    assert c_state["aw1_accepted"] == 0
    assert c_state["aw1_arm"] == "parent"
    for key, value in parent_result["weight_params"].items():
        assert torch.equal(control["weight_params"][key], value)
    result["identity_control_bit_identical"] = "PASS"

    # --- Timing probe at a realistic 4B-like layer size ---------------------
    big_weight, big_wscale, big_calib = make_linear_case(
        device, out_features=4096, in_features=4096, windows=8, tokens=1024,
        seed=719,
    )
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.inference_mode():
        parent.hif4_calibration_and_quantize_weight(big_weight, big_wscale, big_calib)
    if device.type == "cuda":
        torch.cuda.synchronize()
    parent_seconds = time.perf_counter() - t0
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.inference_mode():
        big_result = candidate.hif4_calibration_and_quantize_weight(
            big_weight, big_wscale, big_calib
        )
    if device.type == "cuda":
        torch.cuda.synchronize()
    candidate_seconds = time.perf_counter() - t0
    result["timing_probe_4096x4096_8x1024"] = {
        "parent_seconds": parent_seconds,
        "candidate_seconds": candidate_seconds,
        "aw1_overhead_seconds": candidate_seconds - parent_seconds,
        "aw1_arm": big_result["activation_state"].get("aw1_arm"),
        "aw1_loss_parent": big_result["activation_state"].get("aw1_loss_parent"),
        "aw1_loss_candidate": big_result["activation_state"].get("aw1_loss_candidate"),
        "aw1_gain_abs_dev_mean": big_result["activation_state"].get(
            "aw1_gain_abs_dev_mean"
        ),
    }

    # --- Repo-free single-file import check (six public APIs) ----------------
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        shutil.copy(HERE / "candidate" / "solution.py", tmp_path / "solution.py")
        code = (
            "import sys; sys.path.insert(0, r'%s'); " % str(tmp_path).replace("\\", "/")
            + "import solution; "
            "apis = ['hif4_calibration_and_quantize_weight',"
            "'hif4_dynamic_quantize_activation','hif4_calibration_attention',"
            "'hif4_dynamic_quantize_q','hif4_dynamic_quantize_k',"
            "'hif4_dynamic_quantize_v']; "
            "assert all(callable(getattr(solution, a)) for a in apis); "
            "print('six-api-ok')"
        )
        proc = subprocess.run(
            [sys.executable, "-I", "-c", code],
            cwd=tmp,
            capture_output=True,
            text=True,
            timeout=300,
        )
        assert proc.returncode == 0, proc.stderr
        assert "six-api-ok" in proc.stdout
    result["standalone_import"] = "PASS"

    (HERE / "verification.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
