"""v236 (A-GR1 rebuilt on v231 root) compact controls.

The A-GR1 mechanism itself was fully controlled in
workbench/full_solution/attention-agr1-general-reciprocal/ (0-step bitwise,
reciprocity, logit invariance, reachability).  This card's increment is only
the parent swap (v230 -> v231 Linear, Attention section unchanged), so the
required checks are:
  1. standalone six-API import of the candidate file (outside repo);
  2. candidate calibration runs the A-GR1 path on the v231 root
     (agr1 audit fields present, attempted=1), validate_state passes;
  3. Linear APIs bitwise = v231 archive parent (the only changed section).
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

Q_HEADS, KV_HEADS, HEAD_DIM, TOKENS, N_WINDOWS = 4, 2, 64, 96, 5
NVFP4_MAGNITUDES = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
FIVE_FIELDS = ("scale_factor", "scale_lv2", "scale_lv3", "sign", "mant")


def load_module(path, name):
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


def params_equal(a, b):
    return all(torch.equal(a[k], b[k]) for k in FIVE_FIELDS)


def main():
    torch.manual_seed(0)
    results = {}

    # 1. standalone import outside the repo tree
    tmpdir = tempfile.mkdtemp(prefix="v235_standalone_")
    try:
        standalone = os.path.join(tmpdir, "candidate_solution.py")
        shutil.copyfile(CAND_SOL, standalone)
        old_cwd = os.getcwd()
        saved_path = list(sys.path)
        sys.path = [p for p in sys.path if os.path.abspath(p or ".") != REPO]
        try:
            os.chdir(tmpdir)
            mod = load_module(standalone, "v235_standalone")
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
        c1 = {"missing_apis": missing, "pass": not missing}
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
    results["control1_standalone_import"] = c1

    # 2. calibration + validate_state + agr1 audit
    sys.path.insert(0, REPO)
    from evaluator.reference_hif4 import validate_hif4_params, validate_state

    cand = load_module(CAND_SOL, "v235_candidate")
    windows = [
        {
            "q": make_nvfp4((TOKENS, Q_HEADS * HEAD_DIM), 100 + w),
            "k": make_nvfp4((TOKENS, KV_HEADS * HEAD_DIM), 200 + w),
            "v": make_nvfp4((TOKENS, KV_HEADS * HEAD_DIM), 300 + w),
        }
        for w in range(N_WINDOWS)
    ]
    states = cand.hif4_calibration_attention(windows, Q_HEADS, KV_HEADS, HEAD_DIM)
    validate_state(states["q_state"])
    validate_state(states["k_state"])
    validate_state(states["v_state"])
    q_quant, q_scale = make_nvfp4((TOKENS, Q_HEADS * HEAD_DIM), 999)
    q_fin = cand.hif4_dynamic_quantize_q(q_quant, q_scale, Q_HEADS, HEAD_DIM, states["q_state"])
    validate_hif4_params(q_fin, (TOKENS, Q_HEADS * HEAD_DIM))
    audit = {
        k: states["q_state"].get(k)
        for k in ("a2_arm", "agr1_arm", "agr1_attempted", "agr1_accepted",
                  "agr1_initial_loss", "agr1_final_loss", "agr1_inverse_error")
    }
    c2 = {
        "state_legal": True,
        "deployed_params_legal": True,
        "audit": audit,
        "pass": audit["agr1_attempted"] == 1,
    }
    results["control2_calibration_and_state"] = c2

    # 3. Linear bitwise vs v231 parent
    parent = load_module(PARENT_SOL, "v235_parent")
    w_quant, w_scale = make_nvfp4((128, 128), 555)
    calib_lin = [make_nvfp4((64, 128), 600 + i) for i in range(2)]
    lin_p = parent.hif4_calibration_and_quantize_weight(w_quant, w_scale, calib_lin)
    lin_c = cand.hif4_calibration_and_quantize_weight(w_quant, w_scale, calib_lin)
    a_quant, a_scale = make_nvfp4((32, 128), 777)
    act_p = parent.hif4_dynamic_quantize_activation(a_quant, a_scale, lin_p["activation_state"])
    act_c = cand.hif4_dynamic_quantize_activation(a_quant, a_scale, lin_c["activation_state"])
    c3 = {
        "linear_weight_bitwise": params_equal(lin_p["weight_params"], lin_c["weight_params"]),
        "linear_activation_bitwise": params_equal(act_p, act_c),
    }
    c3["pass"] = all(c3.values())
    results["control3_linear_bitwise_vs_v231"] = c3

    print("=" * 72)
    for name, res in results.items():
        print(f"[{name}] pass={res.get('pass')}")
        for key, value in res.items():
            if key != "pass":
                print(f"    {key}: {value}")
    overall = all(r.get("pass") for r in results.values())
    print(f"OVERALL: {'PASS' if overall else 'FAIL'}")
    return 0 if overall else 1


if __name__ == "__main__":
    sys.exit(main())
