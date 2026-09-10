"""A-FIX1 local controls (plan 2026-09-10-attention-train-deploy-align).

Small synthetic tensors only.  Verifies:
  1. 0 training steps -> Q/K/V five fields and output bitwise = root;
  2. training-forward consistency: Q_hat/K_hat inside the training loop are
     bitwise the deployed hif4_dynamic_quantize_q/k outputs under the same
     player state (current rotation/center), on the same subsampled tokens;
  3. standalone six-API import (outside repo) + validate_state;
  4. V/Linear control bitwise = root;
  5. records: train loss first/final, gate parent/candidate loss, a2_arm,
     rotation/center diff norm vs root, calibration wall time root vs cand.
"""

from __future__ import annotations

import importlib.util
import os
import shutil
import sys
import tempfile
import time

import torch

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
ROOT_SOL = os.path.join(REPO, "solution.py")
CAND_SOL = os.path.join(os.path.dirname(os.path.abspath(__file__)), "candidate", "solution.py")

Q_HEADS = 4
KV_HEADS = 2
HEAD_DIM = 64
TOKENS = 96
N_WINDOWS = 4

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


def main():
    torch.manual_seed(0)
    root = load_module(ROOT_SOL, "afix1_root_solution")
    cand = load_module(CAND_SOL, "afix1_candidate_solution")
    windows = make_calib_windows()
    q_quant, q_scale = make_nvfp4((TOKENS, Q_HEADS * HEAD_DIM), 999)
    k_quant, k_scale = make_nvfp4((TOKENS, KV_HEADS * HEAD_DIM), 998)
    v_quant, v_scale = make_nvfp4((TOKENS, KV_HEADS * HEAD_DIM), 997)

    results = {}

    # ------------------------------------------------------------------
    # Control 1: zero training steps -> bitwise identical to root.
    # ------------------------------------------------------------------
    root._A2_TRAIN_STEPS = 0
    cand._A2_TRAIN_STEPS = 0
    states_root0 = root.hif4_calibration_attention(windows, Q_HEADS, KV_HEADS, HEAD_DIM)
    states_cand0 = cand.hif4_calibration_attention(windows, Q_HEADS, KV_HEADS, HEAD_DIM)
    root._A2_TRAIN_STEPS = 32
    cand._A2_TRAIN_STEPS = 32
    q_r0 = root.hif4_dynamic_quantize_q(q_quant, q_scale, Q_HEADS, HEAD_DIM, states_root0["q_state"])
    k_r0 = root.hif4_dynamic_quantize_k(k_quant, k_scale, KV_HEADS, HEAD_DIM, states_root0["k_state"])
    v_r0 = root.hif4_dynamic_quantize_v(v_quant, v_scale, KV_HEADS, HEAD_DIM, states_root0["v_state"])
    q_c0 = cand.hif4_dynamic_quantize_q(q_quant, q_scale, Q_HEADS, HEAD_DIM, states_cand0["q_state"])
    k_c0 = cand.hif4_dynamic_quantize_k(k_quant, k_scale, KV_HEADS, HEAD_DIM, states_cand0["k_state"])
    v_c0 = cand.hif4_dynamic_quantize_v(v_quant, v_scale, KV_HEADS, HEAD_DIM, states_cand0["v_state"])
    out_r0 = root._a2_attention_forward(
        root._dequantize_hif4(q_r0)[None].float(), root._dequantize_hif4(k_r0)[None].float(),
        root._dequantize_hif4(v_r0)[None].float(), Q_HEADS, KV_HEADS, HEAD_DIM,
    )
    out_c0 = cand._a2_attention_forward(
        cand._dequantize_hif4(q_c0)[None].float(), cand._dequantize_hif4(k_c0)[None].float(),
        cand._dequantize_hif4(v_c0)[None].float(), Q_HEADS, KV_HEADS, HEAD_DIM,
    )
    c1 = {
        "same_arm": states_root0["q_state"].get("a2_arm") == states_cand0["q_state"].get("a2_arm"),
        "q_bitwise": params_equal(q_r0, q_c0),
        "k_bitwise": params_equal(k_r0, k_c0),
        "v_bitwise": params_equal(v_r0, v_c0),
        "final_output_bitwise": bool(torch.equal(out_r0, out_c0)),
    }
    c1["pass"] = all(c1.values())
    results["control1_zero_step_bitwise"] = c1

    # ------------------------------------------------------------------
    # Control 2: training-forward == deployed API (bitwise), same player
    # state and same subsampled tokens.  Recording wrappers capture every
    # dynamic Q/K call made inside the training loop.
    # ------------------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    states_v189 = cand._V189_CALIBRATION_ATTENTION(windows, Q_HEADS, KV_HEADS, HEAD_DIM)
    dense_windows = [
        {
            "q": cand._dequantize_nvfp4_float32(*item["q"]).to(torch.float32),
            "k": cand._dequantize_nvfp4_float32(*item["k"]).to(torch.float32),
            "v": cand._dequantize_nvfp4_float32(*item["v"]).to(torch.float32),
        }
        for item in windows
    ]
    recorded = {"q": [], "k": []}
    orig_q = cand.hif4_dynamic_quantize_q
    orig_k = cand.hif4_dynamic_quantize_k

    def rec_q(*args):
        out = orig_q(*args)
        recorded["q"].append((args, out))
        return out

    def rec_k(*args):
        out = orig_k(*args)
        recorded["k"].append((args, out))
        return out

    cand.hif4_dynamic_quantize_q = rec_q
    cand.hif4_dynamic_quantize_k = rec_k
    cand._A2_TRAIN_STEPS = 2
    try:
        rotation2, info2, center2 = cand._a2_train_rotation(
            dense_windows, Q_HEADS, KV_HEADS, HEAD_DIM, device,
            windows, states_v189,
        )
    finally:
        cand.hif4_dynamic_quantize_q = orig_q
        cand.hif4_dynamic_quantize_k = orig_k
        cand._A2_TRAIN_STEPS = 32

    n_windows = len(dense_windows)
    # expected subsample replication (root _a2_even_indices logic)
    kv_index = cand._a2_even_indices(TOKENS, cand._A2_MAX_KV_TOKENS, device)
    q_cap = min(int(kv_index.numel()), TOKENS)
    q_index = cand._a2_even_indices(q_cap, cand._A2_MAX_Q_TOKENS, device)
    q_rows = kv_index.index_select(0, q_index)

    per_step_ok = True
    direct_bitwise_ok = True
    subsample_ok = True
    shapes_ok = True
    rots = []
    for kind, orig in (("q", orig_q), ("k", orig_k)):
        calls = recorded[kind]
        if len(calls) != 2 * n_windows:
            per_step_ok = False
        for args, out in calls:
            quant_a, scale_a, nh, hd, st = args
            direct = orig(quant_a, scale_a, nh, hd, st)
            if not params_equal(out, direct):
                direct_bitwise_ok = False
            rot = st.get("learned_rotation")
            if not torch.is_tensor(rot) or tuple(rot.shape) != (KV_HEADS, HEAD_DIM, HEAD_DIM):
                shapes_ok = False
            if kind == "k" and not torch.is_tensor(st.get("learned_center")):
                shapes_ok = False
            dense_from_raw = cand._dequantize_nvfp4_float32(quant_a.cpu(), scale_a.cpu()).float()
            if kind == "q":
                expected = dense_windows[0]["q"].index_select(0, q_rows.cpu())
            else:
                expected = dense_windows[0]["k"].index_select(0, kv_index.cpu())
            # calls interleave windows; check membership against any window
            if not any(
                torch.equal(
                    dense_from_raw,
                    w["q" if kind == "q" else "k"].index_select(
                        0, (q_rows if kind == "q" else kv_index).cpu()
                    ),
                )
                for w in dense_windows
            ):
                subsample_ok = False
            if kind == "q":
                rots.append(rot.cpu().clone())
    rotation_moves = any(not torch.equal(rots[0], r) for r in rots[1:])
    c2 = {
        "recorded_q_calls": len(recorded["q"]),
        "recorded_k_calls": len(recorded["k"]),
        "expected_calls_2steps_x_windows": 2 * n_windows,
        "call_count_ok": per_step_ok,
        "direct_call_bitwise": direct_bitwise_ok,
        "player_state_shapes_ok": shapes_ok,
        "subsampled_tokens_match": subsample_ok,
        "rotation_updates_between_calls": rotation_moves,
        "train_loss_first_final": (info2["first_train_loss"], info2["final_train_loss"]),
    }
    c2["pass"] = all([
        per_step_ok, direct_bitwise_ok, shapes_ok, subsample_ok, rotation_moves,
    ])
    results["control2_forward_consistency"] = c2

    # ------------------------------------------------------------------
    # Control 3: standalone import + validate_state on real calibration.
    # ------------------------------------------------------------------
    tmpdir = tempfile.mkdtemp(prefix="afix1_standalone_")
    try:
        standalone = os.path.join(tmpdir, "candidate_solution.py")
        shutil.copyfile(CAND_SOL, standalone)
        old_cwd = os.getcwd()
        saved_path = list(sys.path)
        sys.path = [p for p in sys.path if os.path.abspath(p or ".") != REPO]
        try:
            os.chdir(tmpdir)
            mod = load_module(standalone, "afix1_standalone_candidate")
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

    t0 = time.perf_counter()
    states_cand = cand.hif4_calibration_attention(windows, Q_HEADS, KV_HEADS, HEAD_DIM)
    t_cand = time.perf_counter() - t0
    t0 = time.perf_counter()
    states_root = root.hif4_calibration_attention(windows, Q_HEADS, KV_HEADS, HEAD_DIM)
    t_root = time.perf_counter() - t0
    validate_state(states_cand["q_state"])
    validate_state(states_cand["k_state"])
    validate_state(states_cand["v_state"])
    q_fin = cand.hif4_dynamic_quantize_q(q_quant, q_scale, Q_HEADS, HEAD_DIM, states_cand["q_state"])
    k_fin = cand.hif4_dynamic_quantize_k(k_quant, k_scale, KV_HEADS, HEAD_DIM, states_cand["k_state"])
    validate_hif4_params(q_fin, (TOKENS, Q_HEADS * HEAD_DIM))
    validate_hif4_params(k_fin, (TOKENS, KV_HEADS * HEAD_DIM))
    c3b = {"state_legal": True, "deployed_params_legal": True}
    results["control3b_validate_state"] = {**c3b, "pass": True}

    # ------------------------------------------------------------------
    # Control 4: V control + Linear control (bitwise vs root).
    # ------------------------------------------------------------------
    v_cand = cand.hif4_dynamic_quantize_v(v_quant, v_scale, KV_HEADS, HEAD_DIM, states_root["v_state"])
    v_root = root.hif4_dynamic_quantize_v(v_quant, v_scale, KV_HEADS, HEAD_DIM, states_root["v_state"])
    w_quant, w_scale = make_nvfp4((128, 128), 555)
    calib_lin = [make_nvfp4((64, 128), 600 + i) for i in range(2)]
    lin_root = root.hif4_calibration_and_quantize_weight(w_quant, w_scale, calib_lin)
    lin_cand = cand.hif4_calibration_and_quantize_weight(w_quant, w_scale, calib_lin)
    a_quant, a_scale = make_nvfp4((32, 128), 777)
    act_root = root.hif4_dynamic_quantize_activation(a_quant, a_scale, lin_root["activation_state"])
    act_cand = cand.hif4_dynamic_quantize_activation(a_quant, a_scale, lin_cand["activation_state"])
    c4 = {
        "v_bitwise": params_equal(v_root, v_cand),
        "linear_weight_bitwise": params_equal(lin_root["weight_params"], lin_cand["weight_params"]),
        "linear_activation_bitwise": params_equal(act_root, act_cand),
    }
    c4["pass"] = all(c4.values())
    results["control4_v_and_linear"] = c4

    # ------------------------------------------------------------------
    # Control 5: records (audit, rotation/center diff vs root, timing).
    # ------------------------------------------------------------------
    def audit_of(st):
        return {
            k: st["q_state"].get(k)
            for k in (
                "a2_arm", "a2_gate_loss_identity", "a2_gate_loss_rotation",
                "a2_first_train_loss", "a2_train_loss", "a2_steps", "a2_ortho_error",
            )
        }

    rot_diff = None
    center_diff = None
    if "learned_rotation" in states_root["q_state"] and "learned_rotation" in states_cand["q_state"]:
        rot_diff = float(
            (states_root["q_state"]["learned_rotation"] - states_cand["q_state"]["learned_rotation"]).norm()
        )
        center_diff = float(
            (states_root["k_state"]["learned_center"] - states_cand["k_state"]["learned_center"]).norm()
        )
    c5 = {
        "root_audit": audit_of(states_root),
        "cand_audit": audit_of(states_cand),
        "rotation_diff_frobenius": rot_diff,
        "center_diff_frobenius": center_diff,
        "calib_seconds_root": round(t_root, 3),
        "calib_seconds_candidate": round(t_cand, 3),
        "device": device.type,
    }
    c5["pass"] = True
    results["control5_records"] = c5

    print("=" * 72)
    for name, res in results.items():
        print(f"[{name}] pass={res.get('pass')}")
        for key, value in res.items():
            if key != "pass":
                print(f"    {key}: {value}")
    print("=" * 72)
    overall = all(r.get("pass") for r in results.values())
    print(f"OVERALL: {'PASS' if overall else 'FAIL'}")
    return 0 if overall else 1


if __name__ == "__main__":
    sys.exit(main())
