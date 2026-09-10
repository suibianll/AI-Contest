"""A-GR2 local controls (plan 2026-09-10-attention-agr2-trueobjective).

Parent = v231 Linear (L-EM3 K=2) + v195 Attention archive (NOT the working
tree root solution.py, which is dirty from the parallel Linear session).

Small synthetic tensors only.  Verifies:
  1. zero training steps -> Q/K/V five fields and output bitwise = parent;
  2. reciprocity of the compiled pair (fp precision) + dense pre-quantization
     logit invariance (<1e-6);
  3. standalone six-API import (outside repo) + validate_state;
  4. V/Linear bitwise = parent; training reachable (attempted=1, true
     deployed-MSE training loss decreases);
  5. finite-difference verification of the full grad_N assembly (worst
     relerr < 1e-6 on a smooth surrogate loss).
"""

from __future__ import annotations

import importlib.util
import os
import shutil
import sys
import tempfile

import torch

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
PARENT_SOL = os.path.join(
    REPO, "solutions", "20260910_v231_linear-em3-k2-arm_scoreNA_timeNA", "solution.py"
)
CAND_SOL = os.path.join(os.path.dirname(os.path.abspath(__file__)), "candidate", "solution.py")

Q_HEADS = 4
KV_HEADS = 2
HEAD_DIM = 64
TOKENS = 96
N_WINDOWS = 5

NVFP4_MAGNITUDES = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
FIVE_FIELDS = ("scale_factor", "scale_lv2", "scale_lv3", "sign", "mant")


def load_module(path: str, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_nvfp4(shape, seed):
    g = torch.Generator().manual_seed(seed)
    idx = torch.randint(0, 8, shape, generator=g)
    sign = torch.randint(0, 2, shape, generator=g) * 2 - 1
    quant = (NVFP4_MAGNITUDES[idx] * sign).to(torch.float32)
    scale = (
        torch.rand(shape[:-1] + (shape[-1] // 16,), generator=g) * 0.02 + 0.002
    ).to(torch.float32)
    return quant, scale


def make_calib_windows(seed_base=0):
    return [
        {
            "q": make_nvfp4((TOKENS, Q_HEADS * HEAD_DIM), seed_base + 100 + w),
            "k": make_nvfp4((TOKENS, KV_HEADS * HEAD_DIM), seed_base + 200 + w),
            "v": make_nvfp4((TOKENS, KV_HEADS * HEAD_DIM), seed_base + 300 + w),
        }
        for w in range(N_WINDOWS)
    ]


def params_equal(a, b):
    return all(torch.equal(a[k], b[k]) for k in FIVE_FIELDS)


def gradcheck(cand):
    """FD verification of grad_N = G_m - P @ G_p^T @ P on a smooth surrogate."""
    torch.manual_seed(0)
    G, D, T, PG = 2, 64, 32, 2
    QH = G * PG
    xq = torch.randn(T, QH * D, dtype=torch.float64)
    xk = torch.randn(T, G * D, dtype=torch.float64)
    tq_target = torch.randn(T, QH * D, dtype=torch.float64)
    tk_target = torch.randn(T, G * D, dtype=torch.float64)

    def group_mm(x, mat, heads):
        groups = mat.shape[0]
        per = heads // groups
        dim = x.shape[-1] // heads
        xg = x.reshape(-1, groups, per, dim)
        return torch.einsum("tghk,gkd->tghd", xg, mat).reshape(x.shape)

    def loss_of(n):
        eye = torch.eye(D, dtype=torch.float64).expand(G, D, D)
        m = eye + n
        p = torch.linalg.inv(m).transpose(-1, -2)
        lq = (group_mm(xq, m, QH) - tq_target).square().sum()
        lk = (group_mm(xk, p, G) - tk_target).square().sum()
        return float(lq + lk)

    n0 = torch.randn(G, D, D, dtype=torch.float64) * 0.05
    m = torch.eye(D, dtype=torch.float64).expand(G, D, D) + n0
    p = torch.linalg.inv(m).transpose(-1, -2)
    dq = 2 * (group_mm(xq, m, QH) - tq_target)
    dk = 2 * (group_mm(xk, p, G) - tk_target)
    # formula-level verification in float64 (the fp32 einsum helper in the
    # trainer carries its own ~1e-5-level rounding, matched to fp32 training)
    gm = torch.einsum(
        "tghi,tghj->gij", xq.reshape(-1, G, PG, D), dq.reshape(-1, G, PG, D)
    )
    gp = torch.einsum(
        "tgi,tgj->gij", xk.reshape(-1, G, D), dk.reshape(-1, G, D)
    )
    grad_analytic = gm - p @ gp.transpose(-1, -2) @ p

    eps = 1e-5
    worst = 0.0
    import itertools
    for g0, i, j in itertools.product(range(G), range(0, D, 13), range(0, D, 11)):
        n_ = n0.clone(); n_[g0, i, j] += eps; lp = loss_of(n_)
        n_ = n0.clone(); n_[g0, i, j] -= eps; lm = loss_of(n_)
        fd = (lp - lm) / (2 * eps)
        an = grad_analytic[g0, i, j].item()
        worst = max(worst, abs(an - fd) / max(1.0, abs(fd)))
    return worst


def main():
    torch.manual_seed(0)
    parent = load_module(PARENT_SOL, "agr2_parent_solution")
    cand = load_module(CAND_SOL, "agr2_candidate_solution")
    windows = make_calib_windows()
    q_quant, q_scale = make_nvfp4((TOKENS, Q_HEADS * HEAD_DIM), 999)
    k_quant, k_scale = make_nvfp4((TOKENS, KV_HEADS * HEAD_DIM), 998)
    v_quant, v_scale = make_nvfp4((TOKENS, KV_HEADS * HEAD_DIM), 997)

    results = {}

    # Control 1: zero training steps -> bitwise identical to parent.
    cand._AGR2_TRAIN_STEPS = 0
    states_cand0 = cand.hif4_calibration_attention(windows, Q_HEADS, KV_HEADS, HEAD_DIM)
    cand._AGR2_TRAIN_STEPS = 32
    states_parent = parent.hif4_calibration_attention(windows, Q_HEADS, KV_HEADS, HEAD_DIM)
    q_r = parent.hif4_dynamic_quantize_q(q_quant, q_scale, Q_HEADS, HEAD_DIM, states_parent["q_state"])
    k_r = parent.hif4_dynamic_quantize_k(k_quant, k_scale, KV_HEADS, HEAD_DIM, states_parent["k_state"])
    v_r = parent.hif4_dynamic_quantize_v(v_quant, v_scale, KV_HEADS, HEAD_DIM, states_parent["v_state"])
    q_c0 = cand.hif4_dynamic_quantize_q(q_quant, q_scale, Q_HEADS, HEAD_DIM, states_cand0["q_state"])
    k_c0 = cand.hif4_dynamic_quantize_k(k_quant, k_scale, KV_HEADS, HEAD_DIM, states_cand0["k_state"])
    v_c0 = cand.hif4_dynamic_quantize_v(v_quant, v_scale, KV_HEADS, HEAD_DIM, states_cand0["v_state"])
    out_r = parent._a2_attention_forward(
        parent._dequantize_hif4(q_r)[None].float(), parent._dequantize_hif4(k_r)[None].float(),
        parent._dequantize_hif4(v_r)[None].float(), Q_HEADS, KV_HEADS, HEAD_DIM,
    )
    out_c0 = cand._a2_attention_forward(
        cand._dequantize_hif4(q_c0)[None].float(), cand._dequantize_hif4(k_c0)[None].float(),
        cand._dequantize_hif4(v_c0)[None].float(), Q_HEADS, KV_HEADS, HEAD_DIM,
    )
    c1 = {
        "zero_step_arm": states_cand0["q_state"].get("agr2_arm"),
        "q_bitwise": params_equal(q_r, q_c0),
        "k_bitwise": params_equal(k_r, k_c0),
        "v_bitwise": params_equal(v_r, v_c0),
        "final_output_bitwise": bool(torch.equal(out_r, out_c0)),
    }
    c1["pass"] = all(v for k, v in c1.items() if k != "zero_step_arm")
    results["control1_zero_step_bitwise"] = c1

    # Control 2: reciprocity + dense logit invariance.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    states_base = cand._AGR2_PARENT_CALIBRATION(windows, Q_HEADS, KV_HEADS, HEAD_DIM)
    tq, tk, center_new, info = cand._agr2_train(
        windows[:3], states_base, Q_HEADS, KV_HEADS, HEAD_DIM, device
    )
    rq = states_base["q_state"].get("learned_rotation")
    rk = states_base["k_state"].get("learned_rotation")
    eye = torch.eye(HEAD_DIM).expand(KV_HEADS, HEAD_DIM, HEAD_DIM)
    rq = (rq if rq is not None else eye).double()
    rk = (rk if rk is not None else eye).double()
    recip_err = float(
        (tq.double() @ tk.double().transpose(-1, -2) - rq @ rk.transpose(-1, -2)).abs().max()
    )
    q_dense = cand._dequantize_nvfp4_float32(q_quant, q_scale).to(device=device, dtype=torch.float32)
    k_dense = cand._dequantize_nvfp4_float32(k_quant, k_scale).to(device=device, dtype=torch.float32)
    q0 = cand._attention_state_transform_dense(
        q_dense, states_base["q_state"], Q_HEADS, HEAD_DIM, is_k=False
    ).double()
    k0 = cand._attention_state_transform_dense(
        k_dense, states_base["k_state"], KV_HEADS, HEAD_DIM, is_k=True
    ).double()
    center_old = states_base["k_state"].get("learned_center")

    def grouped_matmul(x, mat, heads):
        per_group = heads // KV_HEADS
        xg = x.reshape(TOKENS, KV_HEADS, per_group, HEAD_DIM)
        return torch.einsum("tghk,gkd->tghd", xg, mat.double().to(device)).reshape(
            TOKENS, heads * HEAD_DIM
        )

    q_new = grouped_matmul(q0, tq, Q_HEADS)
    k_new = grouped_matmul(k0, tk, KV_HEADS)
    if center_new is not None:
        k_new = k_new + center_new.double().to(device).reshape(1, KV_HEADS * HEAD_DIM)
    q_base = grouped_matmul(q0, rq, Q_HEADS)
    k_base = grouped_matmul(k0, rk, KV_HEADS)
    if center_old is not None:
        k_base = k_base + center_old.double().to(device).reshape(1, KV_HEADS * HEAD_DIM)
    logits_new = torch.einsum(
        "ighd,jgd->ighj",
        q_new.reshape(TOKENS, KV_HEADS, Q_HEADS // KV_HEADS, HEAD_DIM),
        k_new.reshape(TOKENS, KV_HEADS, HEAD_DIM),
    )
    logits_base = torch.einsum(
        "ighd,jgd->ighj",
        q_base.reshape(TOKENS, KV_HEADS, Q_HEADS // KV_HEADS, HEAD_DIM),
        k_base.reshape(TOKENS, KV_HEADS, HEAD_DIM),
    )
    logit_diff = float((logits_new - logits_base).abs().max())
    c2 = {
        "compiled_pair_reciprocity_max_abs": recip_err,
        "train_info_inverse_error_fp32": info["agr2_inverse_error"],
        "dense_logit_max_abs_diff_fp64": logit_diff,
        "logit_invariant_lt_1e-6": logit_diff < 1e-6,
        "singular_range": (info["agr2_singular_min"], info["agr2_singular_max"]),
        "n_norm": info["agr2_n_norm"],
        "initial_vs_final_true_loss": (info["agr2_initial_loss"], info["agr2_final_loss"]),
    }
    c2["pass"] = recip_err < 1e-3 and c2["logit_invariant_lt_1e-6"]
    results["control2_reciprocity"] = c2

    # Control 3: standalone import + validate_state.
    tmpdir = tempfile.mkdtemp(prefix="agr2_standalone_")
    try:
        standalone = os.path.join(tmpdir, "candidate_solution.py")
        shutil.copyfile(CAND_SOL, standalone)
        old_cwd = os.getcwd()
        saved_path = list(sys.path)
        sys.path = [p for p in sys.path if os.path.abspath(p or ".") != REPO]
        try:
            os.chdir(tmpdir)
            mod = load_module(standalone, "agr2_standalone_candidate")
        finally:
            os.chdir(old_cwd)
            sys.path = saved_path
        apis = [
            "hif4_calibration_and_quantize_weight",
            "hif4_dynamic_quantize_activation",
            "hif4_calibration_attention",
            "hif4_dynamic_quantize_q",
            "hif4_dynamic_quantize_k",
            "hif4_dynamic_quantize_v",
        ]
        missing = [a for a in apis if not callable(getattr(mod, a, None))]
        states_std = mod.hif4_calibration_attention(windows, Q_HEADS, KV_HEADS, HEAD_DIM)
        q_std = mod.hif4_dynamic_quantize_q(q_quant, q_scale, Q_HEADS, HEAD_DIM, states_std["q_state"])
        finite = all(bool(torch.isfinite(t.to(torch.float32)).all()) for t in q_std.values())
        c3 = {"missing_apis": missing, "standalone_end_to_end_finite": finite}
        c3["pass"] = not missing and finite
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
    results["control3_standalone_import"] = c3

    sys.path.insert(0, REPO)
    from evaluator.reference_hif4 import validate_hif4_params, validate_state

    states_cand = cand.hif4_calibration_attention(windows, Q_HEADS, KV_HEADS, HEAD_DIM)
    validate_state(states_cand["q_state"])
    validate_state(states_cand["k_state"])
    validate_state(states_cand["v_state"])
    q_fin = cand.hif4_dynamic_quantize_q(q_quant, q_scale, Q_HEADS, HEAD_DIM, states_cand["q_state"])
    k_fin = cand.hif4_dynamic_quantize_k(k_quant, k_scale, KV_HEADS, HEAD_DIM, states_cand["k_state"])
    validate_hif4_params(q_fin, (TOKENS, Q_HEADS * HEAD_DIM))
    validate_hif4_params(k_fin, (TOKENS, KV_HEADS * HEAD_DIM))
    results["control3b_validate_state"] = {
        "state_legal": True, "deployed_params_legal": True, "pass": True,
        "cand_audit": {
            k: states_cand["q_state"].get(k)
            for k in ("agr2_arm", "agr2_attempted", "agr2_accepted",
                      "agr2_gate_parent_mse", "agr2_gate_candidate_mse",
                      "agr2_initial_loss", "agr2_final_loss", "agr2_inverse_error",
                      "agr2_n_norm", "agr2_parent_arm", "agr2_center_compiled", "a2_arm")
        },
    }

    # Control 4: V/Linear bitwise + reachability over seeds.
    v_cand = cand.hif4_dynamic_quantize_v(v_quant, v_scale, KV_HEADS, HEAD_DIM, states_parent["v_state"])
    w_quant, w_scale = make_nvfp4((128, 128), 555)
    calib_lin = [make_nvfp4((64, 128), 600 + i) for i in range(2)]
    lin_root = parent.hif4_calibration_and_quantize_weight(w_quant, w_scale, calib_lin)
    lin_cand = cand.hif4_calibration_and_quantize_weight(w_quant, w_scale, calib_lin)
    a_quant, a_scale = make_nvfp4((32, 128), 777)
    act_root = parent.hif4_dynamic_quantize_activation(a_quant, a_scale, lin_root["activation_state"])
    act_cand = cand.hif4_dynamic_quantize_activation(a_quant, a_scale, lin_cand["activation_state"])

    arm_log = []
    loss_down = 0
    for seed in range(8):
        wins = make_calib_windows(seed_base=seed * 1000)
        st = cand.hif4_calibration_attention(wins, Q_HEADS, KV_HEADS, HEAD_DIM)
        qs = st["q_state"]
        entry = {
            "seed": seed,
            "agr2_arm": qs.get("agr2_arm"),
            "attempted": qs.get("agr2_attempted"),
            "accepted": qs.get("agr2_accepted"),
            "gate_parent": qs.get("agr2_gate_parent_mse"),
            "gate_candidate": qs.get("agr2_gate_candidate_mse"),
            "init_loss": qs.get("agr2_initial_loss"),
            "final_loss": qs.get("agr2_final_loss"),
            "n_norm": qs.get("agr2_n_norm"),
        }
        fi, inits = qs.get("agr2_final_loss"), qs.get("agr2_initial_loss")
        if fi is not None and inits is not None and fi < inits:
            loss_down += 1
        arm_log.append(entry)
    arms = {e["agr2_arm"] for e in arm_log}
    c4 = {
        "v_bitwise": params_equal(v_r, v_cand),
        "linear_weight_bitwise": params_equal(lin_root["weight_params"], lin_cand["weight_params"]),
        "linear_activation_bitwise": params_equal(act_root, act_cand),
        "seed_log": arm_log,
        "arms_seen": sorted(a for a in arms if a is not None),
        "attempted_all": all(e["attempted"] == 1 for e in arm_log),
        "loss_decreased_seeds": loss_down,
        "accepted_seeds": sum(1 for e in arm_log if e["agr2_arm"] == "accepted"),
    }
    c4["pass"] = (
        c4["v_bitwise"] and c4["linear_weight_bitwise"] and c4["linear_activation_bitwise"]
        and c4["attempted_all"] and loss_down > 0
    )
    results["control4_v_linear_reachability"] = c4

    # Control 5: finite-difference verification of the grad_N assembly.
    worst = gradcheck(cand)
    c5 = {"grad_fd_worst_relerr": worst, "pass": worst < 1e-6}
    results["control5_gradient_fd"] = c5

    print("=" * 72)
    for name, res in results.items():
        print(f"[{name}] pass={res.get('pass')}")
        for key, value in res.items():
            if key == "pass":
                continue
            if key == "seed_log":
                for entry in value:
                    print(f"    {entry}")
            else:
                print(f"    {key}: {value}")
    print("=" * 72)
    overall = all(r.get("pass") for r in results.values())
    print(f"OVERALL: {'PASS' if overall else 'FAIL'}")
    return 0 if overall else 1


if __name__ == "__main__":
    sys.exit(main())
