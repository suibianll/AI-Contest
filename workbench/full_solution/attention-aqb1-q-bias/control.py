"""A-QB1 local controls (plan 2026-09-10-attention-qk-logit-bias, step 2-4).

Small synthetic tensors only.  Compares the root solution against the A-QB1
candidate for:
  1. b_q = 0 (or absent) -> Q/K/V five fields and final output bitwise = root;
  2. synthetic non-zero b_q -> Q five fields and output change, K/V/Linear
     bitwise unchanged;
  3. standalone six-API import of the single candidate file (outside repo);
  4. validate_state on candidate-calibrated states + training/gate records
     (||b_q||, nonzero channels, gate parent/candidate loss, arm);
  5. V control and Linear control (bitwise identity with root).
A seed loop additionally exercises both gate arms (accepted / reverted) and
verifies the accepted state end-to-end.
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


def make_nvfp4(shape, seed, channel_gain=None):
    g = torch.Generator().manual_seed(seed)
    idx = torch.randint(0, 8, shape, generator=g)
    sign = torch.randint(0, 2, shape, generator=g) * 2 - 1
    quant = (NVFP4_MAGNITUDES[idx] * sign).to(torch.float32)
    if channel_gain is not None:
        quant = quant * channel_gain
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
    root = load_module(ROOT_SOL, "aqb1_root_solution")
    cand = load_module(CAND_SOL, "aqb1_candidate_solution")
    windows = make_calib_windows()
    q_quant, q_scale = make_nvfp4((TOKENS, Q_HEADS * HEAD_DIM), 999)
    k_quant, k_scale = make_nvfp4((TOKENS, KV_HEADS * HEAD_DIM), 998)
    v_quant, v_scale = make_nvfp4((TOKENS, KV_HEADS * HEAD_DIM), 997)

    results = {}

    # ------------------------------------------------------------------
    # Control 1: b_q = 0 (or absent) -> bitwise identical to root.
    # ------------------------------------------------------------------
    states_root = root.hif4_calibration_attention(windows, Q_HEADS, KV_HEADS, HEAD_DIM)
    q_root = root.hif4_dynamic_quantize_q(q_quant, q_scale, Q_HEADS, HEAD_DIM, states_root["q_state"])
    k_root = root.hif4_dynamic_quantize_k(k_quant, k_scale, KV_HEADS, HEAD_DIM, states_root["k_state"])
    v_root = root.hif4_dynamic_quantize_v(v_quant, v_scale, KV_HEADS, HEAD_DIM, states_root["v_state"])
    q_cand_nokey = cand.hif4_dynamic_quantize_q(q_quant, q_scale, Q_HEADS, HEAD_DIM, states_root["q_state"])
    k_cand_nokey = cand.hif4_dynamic_quantize_k(k_quant, k_scale, KV_HEADS, HEAD_DIM, states_root["k_state"])

    zero_b = torch.zeros(Q_HEADS, HEAD_DIM, dtype=torch.float32)
    q_state_zero = dict(states_root["q_state"], learned_q_bias=zero_b)
    q_cand_zero = cand.hif4_dynamic_quantize_q(q_quant, q_scale, Q_HEADS, HEAD_DIM, q_state_zero)

    out_root = root._a2_attention_forward(
        root._dequantize_hif4(q_root)[None].float(),
        root._dequantize_hif4(k_root)[None].float(),
        root._dequantize_hif4(v_root)[None].float(),
        Q_HEADS, KV_HEADS, HEAD_DIM,
    )
    out_cand_zero = cand._a2_attention_forward(
        cand._dequantize_hif4(q_cand_zero)[None].float(),
        cand._dequantize_hif4(k_root)[None].float(),
        cand._dequantize_hif4(v_root)[None].float(),
        Q_HEADS, KV_HEADS, HEAD_DIM,
    )
    c1 = {
        "q_nokey_bitwise": params_equal(q_root, q_cand_nokey),
        "k_nokey_bitwise": params_equal(k_root, k_cand_nokey),
        "q_zero_b_bitwise": params_equal(q_root, q_cand_zero),
        "final_output_zero_b_bitwise": bool(torch.equal(out_root, out_cand_zero)),
    }
    c1["pass"] = all(c1.values())
    results["control1_b0_bitwise"] = c1

    # ------------------------------------------------------------------
    # Control 2: synthetic non-zero b_q -> Q fields/output change, K/V
    # bitwise unchanged.
    # ------------------------------------------------------------------
    synth_b = torch.randn(Q_HEADS, HEAD_DIM, generator=torch.Generator().manual_seed(7)) * 0.5
    q_state_b = dict(states_root["q_state"], learned_q_bias=synth_b)
    q_cand_b = cand.hif4_dynamic_quantize_q(q_quant, q_scale, Q_HEADS, HEAD_DIM, q_state_b)
    k_cand_b = cand.hif4_dynamic_quantize_k(k_quant, k_scale, KV_HEADS, HEAD_DIM, states_root["k_state"])
    v_cand_b = cand.hif4_dynamic_quantize_v(v_quant, v_scale, KV_HEADS, HEAD_DIM, states_root["v_state"])
    q_changed = params_changed_count(q_root, q_cand_b)
    out_cand_b = cand._a2_attention_forward(
        cand._dequantize_hif4(q_cand_b)[None].float(),
        cand._dequantize_hif4(k_root)[None].float(),
        cand._dequantize_hif4(v_root)[None].float(),
        Q_HEADS, KV_HEADS, HEAD_DIM,
    )
    output_max_abs_diff = float((out_root - out_cand_b).abs().max())
    c2 = {
        "q_field_changed_counts": q_changed,
        "q_fields_changed": sum(q_changed.values()) > 0,
        "output_max_abs_diff": output_max_abs_diff,
        "output_changed": output_max_abs_diff > 0.0,
        "k_bitwise_unchanged": params_equal(k_root, k_cand_b),
        "v_bitwise_unchanged": params_equal(v_root, v_cand_b),
    }
    c2["pass"] = (
        c2["q_fields_changed"] and c2["output_changed"]
        and c2["k_bitwise_unchanged"] and c2["v_bitwise_unchanged"]
    )
    results["control2_synthetic_bias"] = c2

    # ------------------------------------------------------------------
    # Control 3: standalone six-API import outside the repo tree.
    # ------------------------------------------------------------------
    tmpdir = tempfile.mkdtemp(prefix="aqb1_standalone_")
    try:
        standalone = os.path.join(tmpdir, "candidate_solution.py")
        shutil.copyfile(CAND_SOL, standalone)
        old_cwd = os.getcwd()
        saved_path = list(sys.path)
        sys.path = [p for p in sys.path if os.path.abspath(p or ".") != REPO]
        try:
            os.chdir(tmpdir)
            mod = load_module(standalone, "aqb1_standalone_candidate")
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
            "aqb1_arm",
            "aqb1_gate_loss_parent",
            "aqb1_gate_loss_candidate",
            "aqb1_steps",
            "aqb1_windows",
            "aqb1_train_loss",
            "aqb1_bias_l2",
            "aqb1_bias_max_abs",
            "aqb1_bias_nonzero",
        )
    }
    q_final = cand.hif4_dynamic_quantize_q(q_quant, q_scale, Q_HEADS, HEAD_DIM, states_cand["q_state"])
    k_final = cand.hif4_dynamic_quantize_k(k_quant, k_scale, KV_HEADS, HEAD_DIM, states_cand["k_state"])
    validate_hif4_params(q_final, (TOKENS, Q_HEADS * HEAD_DIM))
    validate_hif4_params(k_final, (TOKENS, KV_HEADS * HEAD_DIM))
    lb = states_cand["q_state"].get("learned_q_bias")
    c4 = {
        "audit": audit,
        "state_legal": True,
        "deployed_params_legal": True,
        "learned_q_bias_in_state": torch.is_tensor(lb),
        "learned_q_bias_cpu_f32": (
            bool(torch.is_tensor(lb) and lb.device.type == "cpu" and lb.dtype == torch.float32)
            if lb is not None else None
        ),
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
    # Control 6: gate arm exercise over seeds (accepted and reverted).
    # ------------------------------------------------------------------
    arm_log = []
    for seed in range(8):
        wins = make_calib_windows(seed_base=seed * 1000)
        st = cand.hif4_calibration_attention(wins, Q_HEADS, KV_HEADS, HEAD_DIM)
        qs = st["q_state"]
        entry = {
            "seed": seed,
            "aqb1_arm": qs.get("aqb1_arm"),
            "gate_parent": qs.get("aqb1_gate_loss_parent"),
            "gate_candidate": qs.get("aqb1_gate_loss_candidate"),
            "bias_l2": qs.get("aqb1_bias_l2"),
            "bias_nonzero": qs.get("aqb1_bias_nonzero"),
            "bias_max_abs": qs.get("aqb1_bias_max_abs"),
            "windows": qs.get("aqb1_windows"),
        }
        if qs.get("aqb1_arm") == "q-bias":
            lb = qs["learned_q_bias"]
            validate_state(qs)
            entry["bias_shape"] = list(lb.shape)
            entry["bias_cpu_f32"] = bool(lb.device.type == "cpu" and lb.dtype == torch.float32)
            entry["bias_finite"] = bool(torch.isfinite(lb).all())
            p = cand.hif4_dynamic_quantize_q(q_quant, q_scale, Q_HEADS, HEAD_DIM, qs)
            entry["dynamic_q_finite"] = all(
                bool(torch.isfinite(t.float()).all()) for t in p.values()
            )
        arm_log.append(entry)
    arms = {e["aqb1_arm"] for e in arm_log}
    c6 = {
        "log": arm_log,
        "arms_seen": sorted(a for a in arms if a is not None),
        "all_windows_5": all(e["windows"] == N_WINDOWS for e in arm_log),
    }
    # Deterministic revert-path exercise: patch the trainer (test-side only)
    # to return a deliberately harmful bias; the real gate code must reject.
    bad_bias = torch.full((Q_HEADS, HEAD_DIM), 50.0, dtype=torch.float32)
    orig_trainer = cand._aqb1_train_q_bias
    cand._aqb1_train_q_bias = lambda *a, **k: (bad_bias, {
        "steps": 32, "final_train_loss": 0.0, "windows": N_WINDOWS,
        "bias_l2": float(bad_bias.norm()), "bias_max_abs": 50.0,
        "bias_nonzero": Q_HEADS * HEAD_DIM,
    })
    try:
        st_bad = cand.hif4_calibration_attention(windows, Q_HEADS, KV_HEADS, HEAD_DIM)
    finally:
        cand._aqb1_train_q_bias = orig_trainer
    revert_ok = (
        st_bad["q_state"].get("aqb1_arm") == "parent"
        and "learned_q_bias" not in st_bad["q_state"]
        and st_bad["q_state"].get("aqb1_gate_loss_candidate", 0)
        >= st_bad["q_state"].get("aqb1_gate_loss_parent", 1)
    )
    c6["forced_harmful_bias_reverted"] = revert_ok
    c6["pass"] = c6["all_windows_5"] and "q-bias" in arms and revert_ok
    results["control6_gate_arms_over_seeds"] = c6

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
