"""A-G1 local controls (plan 2026-09-10-attention-joint-affine-gauge, step 2-4).

Runs entirely on small synthetic tensors.  Compares the root solution against
the A-G1 candidate for:
  1. s=0 bitwise identity of Q/K five fields and final attention output;
  2. synthetic non-zero zero-mean s: dense softmax invariance + hard Q/K
     five-field changes (reachability);
  3. standalone six-API import of the single candidate file (outside repo);
  4. training/gate records (||s||, nonzero channels, gate losses) + legal
     state validation via evaluator/reference_hif4.validate_state;
  5. V control and Linear control (bitwise identity with root).
"""

from __future__ import annotations

import importlib.util
import math
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
N_WINDOWS = 4

NVFP4_MAGNITUDES = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])


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


def make_calib_windows():
    windows = []
    for w in range(N_WINDOWS):
        windows.append(
            {
                "q": make_nvfp4((TOKENS, Q_HEADS * HEAD_DIM), 100 + w),
                "k": make_nvfp4((TOKENS, KV_HEADS * HEAD_DIM), 200 + w),
                "v": make_nvfp4((TOKENS, KV_HEADS * HEAD_DIM), 300 + w),
            }
        )
    return windows


def params_equal(a, b):
    return all(torch.equal(a[k], b[k]) for k in ("scale_factor", "scale_lv2", "scale_lv3", "sign", "mant"))


def params_changed_count(a, b):
    return {
        k: int((a[k] != b[k]).sum()) for k in ("scale_factor", "scale_lv2", "scale_lv3", "sign", "mant")
    }


def main():
    torch.manual_seed(0)
    root = load_module(ROOT_SOL, "ag1_root_solution")
    cand = load_module(CAND_SOL, "ag1_candidate_solution")
    windows = make_calib_windows()
    q_quant, q_scale = make_nvfp4((TOKENS, Q_HEADS * HEAD_DIM), 999)
    k_quant, k_scale = make_nvfp4((TOKENS, KV_HEADS * HEAD_DIM), 998)
    v_quant, v_scale = make_nvfp4((TOKENS, KV_HEADS * HEAD_DIM), 997)

    results = {}

    # ------------------------------------------------------------------
    # Control 1: s = 0 (or absent) -> bitwise identical to root.
    # ------------------------------------------------------------------
    states_root = root.hif4_calibration_attention(windows, Q_HEADS, KV_HEADS, HEAD_DIM)
    q_root = root.hif4_dynamic_quantize_q(q_quant, q_scale, Q_HEADS, HEAD_DIM, states_root["q_state"])
    k_root = root.hif4_dynamic_quantize_k(k_quant, k_scale, KV_HEADS, HEAD_DIM, states_root["k_state"])
    q_cand_nokey = cand.hif4_dynamic_quantize_q(q_quant, q_scale, Q_HEADS, HEAD_DIM, states_root["q_state"])
    k_cand_nokey = cand.hif4_dynamic_quantize_k(k_quant, k_scale, KV_HEADS, HEAD_DIM, states_root["k_state"])

    zero_s = torch.zeros(KV_HEADS, HEAD_DIM, dtype=torch.float32)
    q_state_zero = dict(states_root["q_state"], learned_scale=zero_s)
    k_state_zero = dict(states_root["k_state"], learned_scale=zero_s)
    q_cand_zero = cand.hif4_dynamic_quantize_q(q_quant, q_scale, Q_HEADS, HEAD_DIM, q_state_zero)
    k_cand_zero = cand.hif4_dynamic_quantize_k(k_quant, k_scale, KV_HEADS, HEAD_DIM, k_state_zero)

    v_root = root.hif4_dynamic_quantize_v(v_quant, v_scale, KV_HEADS, HEAD_DIM, states_root["v_state"])
    out_root = root._a2_attention_forward(
        root._dequantize_hif4(q_root)[None].float(),
        root._dequantize_hif4(k_root)[None].float(),
        root._dequantize_hif4(v_root)[None].float(),
        Q_HEADS, KV_HEADS, HEAD_DIM,
    )
    v_cand_zero = cand.hif4_dynamic_quantize_v(v_quant, v_scale, KV_HEADS, HEAD_DIM, states_root["v_state"])
    out_cand_zero = cand._a2_attention_forward(
        cand._dequantize_hif4(q_cand_zero)[None].float(),
        cand._dequantize_hif4(k_cand_zero)[None].float(),
        cand._dequantize_hif4(v_cand_zero)[None].float(),
        Q_HEADS, KV_HEADS, HEAD_DIM,
    )
    c1 = {
        "q_nokey_bitwise": params_equal(q_root, q_cand_nokey),
        "k_nokey_bitwise": params_equal(k_root, k_cand_nokey),
        "q_zero_s_bitwise": params_equal(q_root, q_cand_zero),
        "k_zero_s_bitwise": params_equal(k_root, k_cand_zero),
        "final_output_zero_s_bitwise": bool(torch.equal(out_root, out_cand_zero)),
    }
    c1["pass"] = all(c1.values())
    results["control1_s0_bitwise"] = c1

    # ------------------------------------------------------------------
    # Control 2: synthetic non-zero zero-mean s.
    # ------------------------------------------------------------------
    s_line = torch.linspace(-0.5, 0.5, HEAD_DIM, dtype=torch.float32)
    s_line = s_line - s_line.mean()
    synth_s = s_line.unsqueeze(0).repeat(KV_HEADS, 1).contiguous()
    assert float(synth_s.mean(dim=-1).abs().max()) < 1e-7

    q_dense = root._dequantize_nvfp4_float32(q_quant, q_scale).float()
    k_dense = root._dequantize_nvfp4_float32(k_quant, k_scale).float()
    v_dense = root._dequantize_nvfp4_float32(v_quant, v_scale).float()
    rot = cand._a2_hadamard_orthogonal(HEAD_DIM).unsqueeze(0).repeat(KV_HEADS, 1, 1)
    center = torch.randn(KV_HEADS, HEAD_DIM, generator=torch.Generator().manual_seed(7)) * 0.1
    d_scale = torch.exp(synth_s)
    q_t = (
        cand._a2_apply_group_rotation(q_dense, Q_HEADS, rot).reshape(TOKENS, KV_HEADS, Q_HEADS // KV_HEADS, HEAD_DIM)
        * d_scale[None, :, None, :]
    ).reshape(TOKENS, Q_HEADS * HEAD_DIM)
    k_rot = cand._a2_apply_group_rotation(k_dense, KV_HEADS, rot)
    k_t = ((k_rot.reshape(TOKENS, KV_HEADS, HEAD_DIM) + center[None]) / d_scale[None]).reshape(TOKENS, KV_HEADS * HEAD_DIM)
    out_orig = cand._a2_attention_forward(q_dense[None], k_dense[None], v_dense[None], Q_HEADS, KV_HEADS, HEAD_DIM)
    out_trans = cand._a2_attention_forward(q_t[None], k_t[None], v_dense[None], Q_HEADS, KV_HEADS, HEAD_DIM)
    dense_max_abs_diff = float((out_orig - out_trans).abs().max())
    dense_rel = dense_max_abs_diff / float(out_orig.abs().max())

    q_state_s = dict(states_root["q_state"], learned_scale=synth_s)
    k_state_s = dict(states_root["k_state"], learned_scale=synth_s)
    q_cand_s = cand.hif4_dynamic_quantize_q(q_quant, q_scale, Q_HEADS, HEAD_DIM, q_state_s)
    k_cand_s = cand.hif4_dynamic_quantize_k(k_quant, k_scale, KV_HEADS, HEAD_DIM, k_state_s)
    q_changed = params_changed_count(q_root, q_cand_s)
    k_changed = params_changed_count(k_root, k_cand_s)
    c2 = {
        "s_zero_mean_max": float(synth_s.mean(dim=-1).abs().max()),
        "dense_output_max_abs_diff": dense_max_abs_diff,
        "dense_output_rel_diff": dense_rel,
        "q_field_changed_counts": q_changed,
        "k_field_changed_counts": k_changed,
    }
    c2["pass"] = (
        dense_rel < 1e-4
        and sum(q_changed.values()) > 0
        and sum(k_changed.values()) > 0
    )
    results["control2_synthetic_s"] = c2

    # ------------------------------------------------------------------
    # Control 3: standalone six-API import outside the repo tree.
    # ------------------------------------------------------------------
    tmpdir = tempfile.mkdtemp(prefix="ag1_standalone_")
    try:
        standalone = os.path.join(tmpdir, "candidate_solution.py")
        shutil.copyfile(CAND_SOL, standalone)
        old_cwd = os.getcwd()
        saved_path = list(sys.path)
        sys.path = [p for p in sys.path if os.path.abspath(p or ".") != REPO]
        try:
            os.chdir(tmpdir)
            mod = load_module(standalone, "ag1_standalone_candidate")
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
        finite = all(
            bool(torch.isfinite(t.to(torch.float32)).all()) for t in q_std.values()
        )
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
            "a2_gate_loss_identity",
            "a2_gate_loss_rotation",
            "a2_steps",
            "a2_train_loss",
            "a2_scale_l2",
            "a2_scale_max_abs",
            "a2_scale_nonzero",
        )
    }
    q_final = cand.hif4_dynamic_quantize_q(q_quant, q_scale, Q_HEADS, HEAD_DIM, states_cand["q_state"])
    k_final = cand.hif4_dynamic_quantize_k(k_quant, k_scale, KV_HEADS, HEAD_DIM, states_cand["k_state"])
    validate_hif4_params(q_final, (TOKENS, Q_HEADS * HEAD_DIM))
    validate_hif4_params(k_final, (TOKENS, KV_HEADS * HEAD_DIM))
    ls = states_cand["q_state"].get("learned_scale")
    trained_changed = {
        "q": params_changed_count(q_root, q_final),
        "k": params_changed_count(k_root, k_final),
    }
    c4 = {
        "audit": audit,
        "state_legal": True,
        "deployed_params_legal": True,
        "learned_scale_in_state": torch.is_tensor(ls),
        "learned_scale_cpu_f32": bool(torch.is_tensor(ls) and ls.device.type == "cpu" and ls.dtype == torch.float32) if ls is not None else None,
        "trained_vs_root_changed_counts": trained_changed,
    }
    c4["pass"] = True
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
