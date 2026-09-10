"""A-QC1 local controls (plan 2026-09-10-attention-q-mean-center, section 4).

Small synthetic tensors only.  Compares the root solution against the A-QC1
candidate for:
  1. arm off (or flag absent) -> Q/K/V five fields and output bitwise = root;
  2. arm on + non-zero-mean Q input -> Q five fields change, K/V/Linear
     bitwise unchanged; dense softmax output difference recorded as
     mechanism-reachability evidence (NO invariance requirement on Q side);
  3. standalone six-API import of the single candidate file (outside repo);
  4. validate_state + per-layer records (arm, gate parent/candidate loss,
     Q changed count);
  5. V control and Linear control (bitwise identity with root);
  6. gate arms: natural decisions over seeds + deterministic reject path.
"""

from __future__ import annotations

import importlib.util
import os
import shutil
import sys
import tempfile

import torch

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
ROOT_SOL = os.path.join(REPO, "solution.py")
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


def make_nvfp4(shape, seed, positive=False):
    g = torch.Generator().manual_seed(seed)
    idx = torch.randint(0, 8, shape, generator=g)
    if positive:
        quant = NVFP4_MAGNITUDES[idx].to(torch.float32) + 0.5
    else:
        sign = torch.randint(0, 2, shape, generator=g) * 2 - 1
        quant = (NVFP4_MAGNITUDES[idx] * sign).to(torch.float32)
    scale = (
        torch.rand(shape[:-1] + (shape[-1] // 16,), generator=g) * 0.02 + 0.002
    ).to(torch.float32)
    return quant, scale


def make_calib_windows(seed_base=0):
    windows = []
    for w in range(N_WINDOWS):
        windows.append(
            {
                "q": make_nvfp4((TOKENS, Q_HEADS * HEAD_DIM), seed_base + 100 + w),
                "k": make_nvfp4((TOKENS, KV_HEADS * HEAD_DIM), seed_base + 200 + w),
                "v": make_nvfp4((TOKENS, KV_HEADS * HEAD_DIM), seed_base + 300 + w),
            }
        )
    return windows


def params_equal(a, b):
    return all(torch.equal(a[k], b[k]) for k in FIVE_FIELDS)


def params_changed_count(a, b):
    return {k: int((a[k] != b[k]).sum()) for k in FIVE_FIELDS}


def main():
    torch.manual_seed(0)
    root = load_module(ROOT_SOL, "aqc1_root_solution")
    cand = load_module(CAND_SOL, "aqc1_candidate_solution")
    windows = make_calib_windows()
    q_quant, q_scale = make_nvfp4((TOKENS, Q_HEADS * HEAD_DIM), 999, positive=True)
    k_quant, k_scale = make_nvfp4((TOKENS, KV_HEADS * HEAD_DIM), 998)
    v_quant, v_scale = make_nvfp4((TOKENS, KV_HEADS * HEAD_DIM), 997)

    results = {}

    # ------------------------------------------------------------------
    # Control 1: arm off (or flag absent) -> bitwise identical to root.
    # ------------------------------------------------------------------
    states_root = root.hif4_calibration_attention(windows, Q_HEADS, KV_HEADS, HEAD_DIM)
    q_root = root.hif4_dynamic_quantize_q(q_quant, q_scale, Q_HEADS, HEAD_DIM, states_root["q_state"])
    k_root = root.hif4_dynamic_quantize_k(k_quant, k_scale, KV_HEADS, HEAD_DIM, states_root["k_state"])
    v_root = root.hif4_dynamic_quantize_v(v_quant, v_scale, KV_HEADS, HEAD_DIM, states_root["v_state"])
    q_cand_nokey = cand.hif4_dynamic_quantize_q(q_quant, q_scale, Q_HEADS, HEAD_DIM, states_root["q_state"])
    k_cand_nokey = cand.hif4_dynamic_quantize_k(k_quant, k_scale, KV_HEADS, HEAD_DIM, states_root["k_state"])
    v_cand_nokey = cand.hif4_dynamic_quantize_v(v_quant, v_scale, KV_HEADS, HEAD_DIM, states_root["v_state"])
    q_state_off = dict(states_root["q_state"], q_mean_center=0)
    q_cand_off = cand.hif4_dynamic_quantize_q(q_quant, q_scale, Q_HEADS, HEAD_DIM, q_state_off)

    out_root = root._a2_attention_forward(
        root._dequantize_hif4(q_root)[None].float(),
        root._dequantize_hif4(k_root)[None].float(),
        root._dequantize_hif4(v_root)[None].float(),
        Q_HEADS, KV_HEADS, HEAD_DIM,
    )
    out_cand_off = cand._a2_attention_forward(
        cand._dequantize_hif4(q_cand_off)[None].float(),
        cand._dequantize_hif4(k_root)[None].float(),
        cand._dequantize_hif4(v_root)[None].float(),
        Q_HEADS, KV_HEADS, HEAD_DIM,
    )
    c1 = {
        "q_bitwise_flag_absent": params_equal(q_root, q_cand_nokey),
        "k_bitwise": params_equal(k_root, k_cand_nokey),
        "v_bitwise": params_equal(v_root, v_cand_nokey),
        "q_bitwise_flag_off": params_equal(q_root, q_cand_off),
        "final_output_bitwise": bool(torch.equal(out_root, out_cand_off)),
    }
    c1["pass"] = all(c1.values())
    results["control1_arm_off_bitwise"] = c1

    # ------------------------------------------------------------------
    # Control 2: arm on + non-zero-mean Q -> Q fields change, K/V
    # bitwise unchanged, dense softmax output difference recorded (no
    # invariance requirement: Q centering deliberately moves logits).
    # ------------------------------------------------------------------
    q_state_on = dict(states_root["q_state"], q_mean_center=1)
    q_cand_on = cand.hif4_dynamic_quantize_q(q_quant, q_scale, Q_HEADS, HEAD_DIM, q_state_on)
    q_changed = params_changed_count(q_root, q_cand_on)

    q_dense = cand._dequantize_nvfp4_float32(q_quant, q_scale).float()
    k_dense = cand._dequantize_nvfp4_float32(k_quant, k_scale).float()
    v_dense = cand._dequantize_nvfp4_float32(v_quant, v_scale).float()
    q_grouped = q_dense.reshape(TOKENS, Q_HEADS, HEAD_DIM)
    q_centered = (q_grouped - q_grouped.mean(dim=0, keepdim=True)).reshape(q_dense.shape)
    q_mean_norm = float(q_grouped.mean(dim=0).norm())
    out_dense_orig = cand._a2_attention_forward(
        q_dense[None], k_dense[None], v_dense[None], Q_HEADS, KV_HEADS, HEAD_DIM
    )
    out_dense_ctr = cand._a2_attention_forward(
        q_centered[None], k_dense[None], v_dense[None], Q_HEADS, KV_HEADS, HEAD_DIM
    )
    dense_max_abs_diff = float((out_dense_orig - out_dense_ctr).abs().max())
    dense_rel_diff = dense_max_abs_diff / float(out_dense_orig.abs().max())

    # hard deployed output also changes (mechanism reaches the output)
    out_cand_on = cand._a2_attention_forward(
        cand._dequantize_hif4(q_cand_on)[None].float(),
        cand._dequantize_hif4(k_root)[None].float(),
        cand._dequantize_hif4(v_root)[None].float(),
        Q_HEADS, KV_HEADS, HEAD_DIM,
    )
    hard_output_max_abs_diff = float((out_root - out_cand_on).abs().max())

    k_cand_arm = cand.hif4_dynamic_quantize_k(k_quant, k_scale, KV_HEADS, HEAD_DIM, states_root["k_state"])
    v_cand_arm = cand.hif4_dynamic_quantize_v(v_quant, v_scale, KV_HEADS, HEAD_DIM, states_root["v_state"])
    c2 = {
        "q_token_mean_norm_input": q_mean_norm,
        "q_field_changed_counts": q_changed,
        "q_fields_changed": sum(q_changed.values()) > 0,
        "dense_softmax_max_abs_diff": dense_max_abs_diff,
        "dense_softmax_rel_diff": dense_rel_diff,
        "dense_output_changed_expected": dense_max_abs_diff > 0.0,
        "hard_output_max_abs_diff": hard_output_max_abs_diff,
        "hard_output_changed": hard_output_max_abs_diff > 0.0,
        "k_bitwise_unchanged": params_equal(k_root, k_cand_arm),
        "v_bitwise_unchanged": params_equal(v_root, v_cand_arm),
    }
    c2["pass"] = (
        q_mean_norm > 0 and c2["q_fields_changed"] and c2["hard_output_changed"]
        and c2["k_bitwise_unchanged"] and c2["v_bitwise_unchanged"]
    )
    results["control2_arm_on"] = c2

    # ------------------------------------------------------------------
    # Control 3: standalone six-API import outside the repo tree.
    # ------------------------------------------------------------------
    tmpdir = tempfile.mkdtemp(prefix="aqc1_standalone_")
    try:
        standalone = os.path.join(tmpdir, "candidate_solution.py")
        shutil.copyfile(CAND_SOL, standalone)
        old_cwd = os.getcwd()
        saved_path = list(sys.path)
        sys.path = [p for p in sys.path if os.path.abspath(p or ".") != REPO]
        try:
            os.chdir(tmpdir)
            mod = load_module(standalone, "aqc1_standalone_candidate")
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

    # ------------------------------------------------------------------
    # Control 4: candidate calibration records + legal state.
    # ------------------------------------------------------------------
    sys.path.insert(0, REPO)
    from evaluator.reference_hif4 import validate_hif4_params, validate_state

    states_cand = cand.hif4_calibration_attention(windows, Q_HEADS, KV_HEADS, HEAD_DIM)
    validate_state(states_cand["q_state"])
    validate_state(states_cand["k_state"])
    validate_state(states_cand["v_state"])
    audit = {
        k: states_cand["q_state"].get(k)
        for k in (
            "a2_arm",
            "aqc1_arm",
            "aqc1_gate_loss_parent",
            "aqc1_gate_loss_candidate",
            "aqc1_windows",
        )
    }
    q_final = cand.hif4_dynamic_quantize_q(q_quant, q_scale, Q_HEADS, HEAD_DIM, states_cand["q_state"])
    k_final = cand.hif4_dynamic_quantize_k(k_quant, k_scale, KV_HEADS, HEAD_DIM, states_cand["k_state"])
    validate_hif4_params(q_final, (TOKENS, Q_HEADS * HEAD_DIM))
    validate_hif4_params(k_final, (TOKENS, KV_HEADS * HEAD_DIM))
    q_trained_changed = params_changed_count(q_root, q_final)
    flag = states_cand["q_state"].get("q_mean_center")
    c4 = {
        "audit": audit,
        "state_legal": True,
        "deployed_params_legal": True,
        "flag_type": type(flag).__name__ if flag is not None else None,
        "q_changed_counts_vs_parent": q_trained_changed,
        "root_a2_arm_preserved": states_cand["q_state"].get("a2_arm"),
        "root_rotation_frozen": (
            ("learned_rotation" in states_root["q_state"])
            == ("learned_rotation" in states_cand["q_state"])
            and (
                "learned_rotation" not in states_root["q_state"]
                or torch.equal(states_root["q_state"]["learned_rotation"], states_cand["q_state"]["learned_rotation"])
            )
        ),
    }
    c4["pass"] = c4["state_legal"] and c4["deployed_params_legal"] and c4["root_rotation_frozen"]
    results["control4_records_and_legal_state"] = c4

    # ------------------------------------------------------------------
    # Control 5: V control + Linear control (bitwise vs root).
    # ------------------------------------------------------------------
    v_cand = cand.hif4_dynamic_quantize_v(v_quant, v_scale, KV_HEADS, HEAD_DIM, states_root["v_state"])
    v_bitwise = params_equal(v_root, v_cand)

    w_quant, w_scale = make_nvfp4((128, 128), 555)
    calib_lin = [make_nvfp4((64, 128), 600 + i) for i in range(2)]
    lin_root = root.hif4_calibration_and_quantize_weight(w_quant, w_scale, calib_lin)
    lin_cand = cand.hif4_calibration_and_quantize_weight(w_quant, w_scale, calib_lin)
    lin_w_bitwise = params_equal(lin_root["weight_params"], lin_cand["weight_params"])
    a_quant, a_scale = make_nvfp4((32, 128), 777)
    act_root = root.hif4_dynamic_quantize_activation(a_quant, a_scale, lin_root["activation_state"])
    act_cand = cand.hif4_dynamic_quantize_activation(a_quant, a_scale, lin_cand["activation_state"])
    lin_a_bitwise = params_equal(act_root, act_cand)
    c5 = {
        "v_bitwise": v_bitwise,
        "linear_weight_bitwise": lin_w_bitwise,
        "linear_activation_bitwise": lin_a_bitwise,
    }
    c5["pass"] = all(c5.values())
    results["control5_v_and_linear"] = c5

    # ------------------------------------------------------------------
    # Control 6: gate arms — natural decisions over seeds + deterministic
    # reject path (forced non-improving candidate through the real gate).
    # ------------------------------------------------------------------
    arm_log = []
    for seed in range(8):
        wins = make_calib_windows(seed_base=seed * 1000)
        st = cand.hif4_calibration_attention(wins, Q_HEADS, KV_HEADS, HEAD_DIM)
        qs = st["q_state"]
        entry = {
            "seed": seed,
            "aqc1_arm": qs.get("aqc1_arm"),
            "gate_parent": qs.get("aqc1_gate_loss_parent"),
            "gate_candidate": qs.get("aqc1_gate_loss_candidate"),
            "windows": qs.get("aqc1_windows"),
            "flag": qs.get("q_mean_center"),
        }
        if qs.get("aqc1_arm") == "center":
            validate_state(qs)
            p = cand.hif4_dynamic_quantize_q(q_quant, q_scale, Q_HEADS, HEAD_DIM, qs)
            entry["dynamic_q_finite"] = all(
                bool(torch.isfinite(t.float()).all()) for t in p.values()
            )
        arm_log.append(entry)
    arms = {e["aqc1_arm"] for e in arm_log}

    orig_gate = cand._aqc1_true_path_gate_losses
    cand._aqc1_true_path_gate_losses = lambda *a, **k: (1.0, 1.0)
    try:
        st_flat = cand.hif4_calibration_attention(windows, Q_HEADS, KV_HEADS, HEAD_DIM)
    finally:
        cand._aqc1_true_path_gate_losses = orig_gate
    reject_ok = (
        st_flat["q_state"].get("aqc1_arm") == "parent"
        and "q_mean_center" not in st_flat["q_state"]
    )
    c6 = {
        "log": arm_log,
        "arms_seen": sorted(a for a in arms if a is not None),
        "all_windows_5": all(e["windows"] == N_WINDOWS for e in arm_log),
        "forced_nonimproving_rejected": reject_ok,
    }
    c6["pass"] = c6["all_windows_5"] and reject_ok
    results["control6_gate_arms"] = c6

    print("=" * 72)
    for name, res in results.items():
        print(f"[{name}] pass={res.get('pass')}")
        for key, value in res.items():
            if key == "pass":
                continue
            if key == "log":
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
