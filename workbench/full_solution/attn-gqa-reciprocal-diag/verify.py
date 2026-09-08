"""Verify v198 attn-gqa-reciprocal-diag on small synthetic tensors.

Geometry: 4 Q heads x 2 KV heads x 64 head_dim x 8 tokens x 5 windows.

Checks:
1. Analytic init + smooth-max refine execute and reduce the range objective;
   identical Q/K channel statistics leave u exactly zero.
2. Reciprocal relation on the deployed continuous tensors: the encoder input
   captured at _dense_to_hif4 satisfies Q' = Q_final * exp(u),
   K' = (K_final + c) * exp(-u) (K-center synchronized compile), and the
   grouped QK^T is numerically identical between parent and candidate.
3. Real training branch + true-path gate accept on outlier data
   (rd_attempted=1, rd_accepted=1, candidate_mse < parent_mse).
4. Degenerate balanced data leaves u == 0 and is rejected to the parent;
   a forced-worse gate also falls back to the untouched parent state.
5. u==0 control path is bitwise identical to the parent encode.
6. Legal state (reference_hif4.validate_state), finite outputs, valid HiF4
   params, and an isolated (python -I) single-file six-API import check.
"""

from pathlib import Path
import hashlib
import importlib.util
import json
import subprocess
import sys

import torch


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "evaluator"))
import reference_hif4 as ref

torch.set_num_threads(1)

Q_HEADS = 4
KV_HEADS = 2
HEAD_DIM = 64
TOKENS = 8
WINDOWS = 5


def load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_windows(device: torch.device, *, q_spike: float = 0.0,
                 k_spike: float = 0.0, balanced: bool = False):
    windows = []
    for index in range(WINDOWS):
        generator = torch.Generator(device=device).manual_seed(19800 + index)
        if balanced:
            # +/-1 entries: per-channel RMS is exactly 1.0 for both Q and K,
            # so the analytic init is exactly u == 0 and stays there.
            q = torch.where(
                torch.rand(TOKENS, Q_HEADS * HEAD_DIM, device=device,
                           generator=generator) < 0.5,
                -1.0, 1.0,
            )
            k = torch.where(
                torch.rand(TOKENS, KV_HEADS * HEAD_DIM, device=device,
                           generator=generator) < 0.5,
                -1.0, 1.0,
            )
            v = torch.randn(TOKENS, KV_HEADS * HEAD_DIM, device=device,
                            generator=generator)
        else:
            q = torch.randn(TOKENS, Q_HEADS * HEAD_DIM, device=device,
                            generator=generator) * 0.02
            k = torch.randn(TOKENS, KV_HEADS * HEAD_DIM, device=device,
                            generator=generator) * 0.02
            v = torch.randn(TOKENS, KV_HEADS * HEAD_DIM, device=device,
                            generator=generator)
            if q_spike > 0.0:
                # Q outlier on channel 3 of every head; K is small there.
                q[:, 3::HEAD_DIM] = (
                    torch.randn(TOKENS, Q_HEADS, device=device,
                                generator=generator) * q_spike
                )
            if k_spike > 0.0:
                # K outlier on channel 27 of every KV head; Q is small there.
                k[:, 27::HEAD_DIM] = (
                    torch.randn(TOKENS, KV_HEADS, device=device,
                                generator=generator) * k_spike
                )
        windows.append({
            "q": (q, torch.ones(TOKENS, Q_HEADS * HEAD_DIM // 16, device=device)),
            "k": (k, torch.ones(TOKENS, KV_HEADS * HEAD_DIM // 16, device=device)),
            "v": (v, torch.ones(TOKENS, KV_HEADS * HEAD_DIM // 16, device=device)),
        })
    return windows


def base_states():
    common = {
        "multiplier": None,
        "permutation": None,
        "importance": None,
        "offsets": torch.tensor((-1, 1, 2, 3, 4), dtype=torch.int8),
        "error_threshold": 1e-7,
        "accept_margin": 0.0,
        "max_refine_ratio": 0.0,
        "max_refine_blocks": 0,
        "version": 2,
    }
    return {
        "q_state": dict(common, num_heads=Q_HEADS, head_dim=HEAD_DIM),
        "k_state": dict(common, center_mode=0, num_heads=KV_HEADS,
                        head_dim=HEAD_DIM),
        "v_state": dict(common, num_heads=KV_HEADS, head_dim=HEAD_DIM),
    }


def rotation_center_states():
    states = base_states()
    generator = torch.Generator().manual_seed(198)
    rotation = torch.linalg.qr(
        torch.randn(KV_HEADS, HEAD_DIM, HEAD_DIM, generator=generator)
    )[0]
    states["q_state"]["learned_rotation"] = rotation.clone()
    states["k_state"]["learned_rotation"] = rotation.clone()
    states["k_state"]["learned_center"] = (
        torch.randn(KV_HEADS, HEAD_DIM, generator=generator) * 0.05
    )
    return states


def capture_encode_dense(module, role: str, window: dict, state: dict):
    heads = Q_HEADS if role == "q" else KV_HEADS
    api = getattr(module, "hif4_dynamic_quantize_" + role)
    captured = []
    original = module._dense_to_hif4

    def spy(dense, **kwargs):
        if not captured:
            captured.append(dense.detach().clone())
        return original(dense, **kwargs)

    module._dense_to_hif4 = spy
    try:
        api(*window[role], heads, HEAD_DIM, state)
    finally:
        module._dense_to_hif4 = original
    assert captured, "encoder did not consume a dense tensor"
    return captured[0]


def grouped_logits(q_dense: torch.Tensor, k_dense: torch.Tensor):
    group = Q_HEADS // KV_HEADS
    qh = q_dense.to(torch.float64).reshape(TOKENS, KV_HEADS, group, HEAD_DIM)
    kh = k_dense.to(torch.float64).reshape(TOKENS, KV_HEADS, HEAD_DIM)
    return torch.einsum("tgjd,tsd->gjts", qh, kh)


def main():
    candidate = load(HERE / "candidate" / "solution.py", "v198_verify_candidate")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    result = {
        "device": str(device),
        "candidate_sha256": hashlib.sha256(
            (HERE / "candidate" / "solution.py").read_bytes()
        ).hexdigest(),
        "parent_sha256": hashlib.sha256(
            (ROOT / "solution.py").read_bytes()
        ).hexdigest(),
    }

    # -- 1. Stage A unit: refine reduces the smooth-max range objective ------
    # One-sided Q outlier with u started at zero: the smooth-max gradient must
    # push u[:, 3] negative and strictly reduce the range objective.
    a = torch.full((KV_HEADS, HEAD_DIM), -3.9, device=device)
    b = torch.full((KV_HEADS, HEAD_DIM), -3.9, device=device)
    a[:, 3] = 0.0    # strong Q-only elevation
    u_start = torch.zeros(KV_HEADS, HEAD_DIM, device=device)
    u1 = candidate._rd_smooth_max_refine(u_start, a, b)
    loss0 = candidate._rd_range_loss(u_start, a, b, 64)
    loss1 = candidate._rd_range_loss(u1, a, b, 64)
    assert loss1 < loss0, (loss0, loss1)
    assert float(u1[:, 3].max()) < 0.0
    assert float((u1 - u_start).abs().max()) > 0.0
    assert float(u1.abs().max()) <= candidate._RD_U_BOUND + 1e-6
    # clamp keeps extreme imbalance inside +/-log(2)
    u_extreme = candidate._rd_smooth_max_refine(
        (0.5 * (b - (a + 6.0))).clamp(min=-candidate._RD_U_BOUND,
                                      max=candidate._RD_U_BOUND),
        a + 6.0, b,
    )
    assert float(u_extreme.abs().max()) <= candidate._RD_U_BOUND + 1e-6
    # identical statistics -> u stays exactly zero
    u_zero = candidate._rd_smooth_max_refine(
        torch.zeros(KV_HEADS, HEAD_DIM, device=device), b, b
    )
    assert float(u_zero.abs().max()) == 0.0
    result["stage_a_refine"] = "PASS"
    result["range_loss_init"] = loss0
    result["range_loss_final"] = loss1

    spike_windows = make_windows(device, q_spike=8.0, k_spike=8.0)
    original_parent = candidate._RD_PARENT_CALIBRATION

    # -- 3. Real training branch + true-path gate accept on outlier data -----
    candidate._RD_PARENT_CALIBRATION = lambda *args, **kwargs: base_states()
    try:
        with torch.inference_mode():
            accepted = candidate.hif4_calibration_attention(
                spike_windows, Q_HEADS, KV_HEADS, HEAD_DIM
            )
    finally:
        candidate._RD_PARENT_CALIBRATION = original_parent
    q_audit = accepted["q_state"]
    assert q_audit["rd_arm"] == "accepted", q_audit
    assert q_audit["rd_attempted"] == 1
    assert q_audit["rd_accepted"] == 1
    assert q_audit["rd_fit_windows"] == WINDOWS - 1
    assert q_audit["rd_candidate_mse"] < q_audit["rd_parent_mse"]
    assert q_audit["rd_u_max_abs"] > 0.5
    assert q_audit["rd_u_nonzero"] > 0
    assert "diag_scale" in accepted["q_state"]
    assert "diag_scale" in accepted["k_state"]
    # V state strictly untouched
    assert accepted["v_state"] is not None
    assert "diag_scale" not in accepted["v_state"]
    result["gate_accept_spike"] = "PASS"
    result["gate_parent_mse"] = q_audit["rd_parent_mse"]
    result["gate_candidate_mse"] = q_audit["rd_candidate_mse"]
    result["rd_u_max_abs"] = q_audit["rd_u_max_abs"]
    result["rd_u_mean_abs"] = q_audit["rd_u_mean_abs"]

    # -- 2. Reciprocal relation + K-center sync on the deployed continuous ---
    rc_states = rotation_center_states()
    u = torch.zeros(KV_HEADS, HEAD_DIM)
    u[:, 3] = -0.4
    u[:, 27] = 0.3
    u[:, 40] = 0.15
    group = Q_HEADS // KV_HEADS
    q_factor = u.exp().repeat_interleave(group, dim=0).reshape(-1)
    k_factor = u.neg().exp().reshape(-1)
    cand_q = dict(rc_states["q_state"], diag_scale=q_factor.clone())
    cand_k = dict(rc_states["k_state"], diag_scale=k_factor.clone())
    window = spike_windows[0]
    with torch.inference_mode():
        q_parent_dense = capture_encode_dense(candidate, "q", window,
                                              rc_states["q_state"])
        k_parent_dense = capture_encode_dense(candidate, "k", window,
                                              rc_states["k_state"])
        q_cand_dense = capture_encode_dense(candidate, "q", window, cand_q)
        k_cand_dense = capture_encode_dense(candidate, "k", window, cand_k)
    torch.testing.assert_close(
        q_cand_dense, q_parent_dense * q_factor.to(q_cand_dense.device),
        rtol=1e-6, atol=1e-6,
    )
    # k_parent_dense already includes the learned center; the candidate encode
    # multiplies the centered tensor by exp(-u), i.e. c is compiled to
    # c * exp(-u) exactly as the reciprocal invariant requires.
    torch.testing.assert_close(
        k_cand_dense, k_parent_dense * k_factor.to(k_cand_dense.device),
        rtol=1e-6, atol=1e-6,
    )
    result["k_center_sync_compile"] = "PASS"
    logits_parent = grouped_logits(q_parent_dense, k_parent_dense)
    logits_cand = grouped_logits(q_cand_dense, k_cand_dense)
    torch.testing.assert_close(logits_cand, logits_parent, rtol=1e-5, atol=1e-7)
    result["reciprocal_qkt_identity_max_err"] = float(
        (logits_cand - logits_parent).abs().max()
    )
    result["reciprocal_qkt_identity"] = "PASS"

    # -- 4a. Degenerate balanced data: u == 0, gate rejects, parent kept -----
    balanced_windows = make_windows(device, balanced=True)
    candidate._RD_PARENT_CALIBRATION = lambda *args, **kwargs: base_states()
    try:
        with torch.inference_mode():
            rejected = candidate.hif4_calibration_attention(
                balanced_windows, Q_HEADS, KV_HEADS, HEAD_DIM
            )
    finally:
        candidate._RD_PARENT_CALIBRATION = original_parent
    r_audit = rejected["q_state"]
    assert r_audit["rd_arm"] == "parent", r_audit
    assert r_audit["rd_attempted"] == 1
    assert r_audit["rd_accepted"] == 0
    assert r_audit["rd_u_max_abs"] == 0.0
    assert r_audit["rd_candidate_mse"] == r_audit["rd_parent_mse"]
    assert "diag_scale" not in rejected["q_state"]
    assert "diag_scale" not in rejected["k_state"]
    result["degenerate_u_zero_parent_fallback"] = "PASS"

    # -- 4b. Forced-worse gate also falls back to the untouched parent -------
    original_mse = candidate._rd_true_output_mse
    calls = {"count": 0}

    def fake_mse(*args, **kwargs):
        calls["count"] += 1
        return 1.0 if calls["count"] % 2 == 1 else 2.0

    candidate._RD_PARENT_CALIBRATION = lambda *args, **kwargs: base_states()
    candidate._rd_true_output_mse = fake_mse
    try:
        with torch.inference_mode():
            forced = candidate.hif4_calibration_attention(
                spike_windows, Q_HEADS, KV_HEADS, HEAD_DIM
            )
    finally:
        candidate._RD_PARENT_CALIBRATION = original_parent
        candidate._rd_true_output_mse = original_mse
    assert forced["q_state"]["rd_arm"] == "parent"
    assert forced["q_state"]["rd_attempted"] == 1
    assert forced["q_state"]["rd_accepted"] == 0
    assert "diag_scale" not in forced["q_state"]
    result["forced_worse_gate_fallback"] = "PASS"

    # -- 5. u == 0 control path is bitwise identical to the parent encode ----
    ones_q = torch.ones(Q_HEADS * HEAD_DIM)
    ones_k = torch.ones(KV_HEADS * HEAD_DIM)
    with torch.inference_mode():
        for role, heads, ones in (
            ("q", Q_HEADS, ones_q),
            ("k", KV_HEADS, ones_k),
        ):
            base = base_states()[role + "_state"]
            api = getattr(candidate, "hif4_dynamic_quantize_" + role)
            p_none = api(*spike_windows[0][role], heads, HEAD_DIM, base)
            p_ones = api(*spike_windows[0][role], heads, HEAD_DIM,
                         dict(base, diag_scale=ones))
            for key in ("scale_factor", "scale_lv2", "scale_lv3", "sign",
                        "mant"):
                assert torch.equal(p_none[key], p_ones[key]), key
    # rejected (parent) states reproduce the parent encode bitwise
    with torch.inference_mode():
        clean = base_states()
        for role, heads in (("q", Q_HEADS), ("k", KV_HEADS), ("v", KV_HEADS)):
            api = getattr(candidate, "hif4_dynamic_quantize_" + role)
            p_clean = api(*spike_windows[0][role], heads, HEAD_DIM,
                          clean[role + "_state"])
            p_rej = api(*spike_windows[0][role], heads, HEAD_DIM,
                        rejected[role + "_state"])
            for key in ("scale_factor", "scale_lv2", "scale_lv3", "sign",
                        "mant"):
                assert torch.equal(p_clean[key], p_rej[key]), (role, key)
    result["zero_u_bitwise_control"] = "PASS"

    # -- 6. Legal state, valid HiF4 params, finite outputs -------------------
    with torch.inference_mode():
        for role, heads in (("q", Q_HEADS), ("k", KV_HEADS), ("v", KV_HEADS)):
            state = accepted[role + "_state"]
            ref.validate_state(state)
            api = getattr(candidate, "hif4_dynamic_quantize_" + role)
            params = api(*spike_windows[0][role], heads, HEAD_DIM, state)
            ref.validate_hif4_params(
                params, spike_windows[0][role][0].shape
            )
            assert bool(torch.isfinite(
                candidate._dequantize_hif4(params)
            ).all())
    result["state_and_output_contract"] = "PASS"

    # -- 7. Isolated single-file import check (python -I) ---------------------
    isolated_dir = HERE / "isolated"
    isolated_dir.mkdir(exist_ok=True)
    isolated_copy = isolated_dir / "solution.py"
    isolated_copy.write_bytes((HERE / "candidate" / "solution.py").read_bytes())
    code = (
        "import importlib.util, sys\n"
        "spec = importlib.util.spec_from_file_location('cand', sys.argv[1])\n"
        "module = importlib.util.module_from_spec(spec)\n"
        "spec.loader.exec_module(module)\n"
        "apis = ['hif4_calibration_and_quantize_weight',"
        " 'hif4_dynamic_quantize_activation', 'hif4_calibration_attention',"
        " 'hif4_dynamic_quantize_q', 'hif4_dynamic_quantize_k',"
        " 'hif4_dynamic_quantize_v']\n"
        "for name in apis:\n"
        "    assert callable(getattr(module, name, None)), name\n"
        "print('isolated import OK')\n"
    )
    completed = subprocess.run(
        [sys.executable, "-I", "-c", code, str(isolated_copy)],
        cwd=str(isolated_dir), capture_output=True, text=True, check=True,
    )
    assert "isolated import OK" in completed.stdout
    result["isolated_import"] = "PASS"

    (HERE / "verification.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
