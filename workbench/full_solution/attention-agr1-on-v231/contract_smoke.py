"""Random-shape contract smoke for the v236 candidate (AGENTS section 4.2).

A-GR1 is a training-class mechanism, so the archived file must survive the six
public APIs on shapes other than the one the card happened to run.

This smoke is DIFFERENTIAL, not absolute, and it is THREE-WAY.  Two inherited
shape behaviours were found by probing and both must not be reported as this
card's regression:

  * qh=6 kv=3 hd=32 raises "q head count does not match calibration state" on
    the v231 ROOT as well (K/V hidden dim 96 is not a multiple of 64).
  * hd=32 leaves A-GR1 at agr1_arm="fallback" with the block's own documented
    message "Residual training requires a 64-divisible head dimension" -- and it
    does so identically in the v234 archive, which holds the same Attention
    code.  It is a graceful fallback that preserves the parent state, and the
    real 4B panel (head_dim 256) never triggers it.

So the test is:
  1. the candidate must accept/reject exactly what the v231 PARENT accepts and
     rejects (no drift in the surrounding contract);
  2. the candidate must reach / decline the A-GR1 branch exactly where the v234
     MECHANISM REFERENCE does, and report the same arm ("parent"/"fallback");
  3. on shapes where the branch is reached, the candidate must produce
     validator-legal state, validator-legal deployed params and finite output.

The reference validator is evaluator/reference_hif4.py, so this is not a
self-consistency test.
"""

from __future__ import annotations

import importlib.util
import os
import sys

import torch

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
CAND_SOL = os.path.join(os.path.dirname(os.path.abspath(__file__)), "candidate", "solution.py")
ROOT_SOL = os.path.join(REPO, "solution.py")
MECH_REF_SOL = os.path.join(
    REPO,
    "solutions",
    "20260910_v234_attention-agr1-general-reciprocal_scoreNA_timeNA",
    "solution.py",
)

NVFP4_MAGNITUDES = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
FIVE_FIELDS = ("scale_factor", "scale_lv2", "scale_lv3", "sign", "mant")

# (q_heads, kv_heads, head_dim, tokens, windows)
SHAPES = [
    (4, 2, 64, 96, 5),    # the shape the card ran with
    (16, 4, 256, 40, 5),  # the 4B panel's real GQA geometry
    (8, 8, 64, 17, 6),    # MHA-like, extra window, odd token count
    (2, 1, 64, 5, 5),     # smallest legal, very short sequence
    (16, 4, 64, 128, 5),  # wider Q/KV ratio
    (4, 2, 128, 64, 5),   # wide head_dim
    (6, 2, 64, 64, 5),    # non-power-of-two head count
    (6, 3, 32, 200, 5),   # ILLEGAL by the parent's contract (KV dim 96, not a 64-multiple)
    (16, 4, 32, 64, 5),   # legal narrow head_dim, for contrast with the line above
]


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
    scale = (torch.rand(shape[:-1] + (shape[-1] // 16,), generator=g) * 0.02 + 0.002).to(
        torch.float32
    )
    return quant, scale


def params_finite(p):
    return all(torch.isfinite(p[k].float()).all().item() for k in FIVE_FIELDS)


def run_shape(mod, q_heads, kv_heads, head_dim, tokens, n_windows, validate_state, validate_hif4_params):
    """Return (outcome, detail). outcome is 'ok' or the exception type name."""
    try:
        windows = [
            {
                "q": make_nvfp4((tokens, q_heads * head_dim), 1000 + w),
                "k": make_nvfp4((tokens, kv_heads * head_dim), 2000 + w),
                "v": make_nvfp4((tokens, kv_heads * head_dim), 3000 + w),
            }
            for w in range(n_windows)
        ]
        states = mod.hif4_calibration_attention(windows, q_heads, kv_heads, head_dim)
        for side in ("q", "k", "v"):
            validate_state(states[f"{side}_state"])

        q_quant, q_scale = make_nvfp4((tokens, q_heads * head_dim), 999)
        k_quant, k_scale = make_nvfp4((tokens, kv_heads * head_dim), 998)
        v_quant, v_scale = make_nvfp4((tokens, kv_heads * head_dim), 997)
        q_fin = mod.hif4_dynamic_quantize_q(q_quant, q_scale, q_heads, head_dim, states["q_state"])
        k_fin = mod.hif4_dynamic_quantize_k(k_quant, k_scale, kv_heads, head_dim, states["k_state"])
        v_fin = mod.hif4_dynamic_quantize_v(v_quant, v_scale, kv_heads, head_dim, states["v_state"])
        validate_hif4_params(q_fin, (tokens, q_heads * head_dim))
        validate_hif4_params(k_fin, (tokens, kv_heads * head_dim))
        validate_hif4_params(v_fin, (tokens, kv_heads * head_dim))
        q_state = states["q_state"]
        detail = {
            "finite": all(params_finite(p) for p in (q_fin, k_fin, v_fin)),
            "agr1_attempted": q_state.get("agr1_attempted"),
            "agr1_accepted": q_state.get("agr1_accepted"),
            "agr1_arm": q_state.get("agr1_arm"),
            "agr1_error": q_state.get("agr1_error"),
            "inverse_error": q_state.get("agr1_inverse_error"),
        }
        return "ok", detail
    except Exception as exc:  # noqa: BLE001 - the outcome is the finding
        return type(exc).__name__, str(exc)[:70]


def main():
    torch.manual_seed(0)
    sys.path.insert(0, REPO)
    from evaluator.reference_hif4 import validate_hif4_params, validate_state

    root = load_module(ROOT_SOL, "v236_contract_root")
    cand = load_module(CAND_SOL, "v236_contract_candidate")
    ref = load_module(MECH_REF_SOL, "v236_contract_mechanism_ref")

    print("=" * 118)
    print(f"{'shape':<34}{'parent':<20}{'mech-ref(v234)':<26}{'candidate':<26}verdict")
    ok = True
    for shape in SHAPES:
        label = "qh=%d kv=%d hd=%d tok=%d win=%d" % shape
        r_out, _ = run_shape(root, *shape, validate_state, validate_hif4_params)
        c_out, c_detail = run_shape(cand, *shape, validate_state, validate_hif4_params)
        f_out, f_detail = run_shape(ref, *shape, validate_state, validate_hif4_params)

        ref_arm = f_detail.get("agr1_arm") if f_out == "ok" else f_out
        cand_arm = c_detail.get("agr1_arm") if c_out == "ok" else c_out

        if r_out != c_out:
            verdict = "FAIL: candidate diverges from parent on accept/reject"
            ok = False
        elif f_out != c_out:
            verdict = "FAIL: candidate diverges from mechanism reference on accept/reject"
            ok = False
        elif ref_arm != cand_arm:
            verdict = f"FAIL: arm differs from mechanism reference ({ref_arm} vs {cand_arm})"
            ok = False
        elif r_out != "ok":
            verdict = f"ok: parent and reference both reject ({r_out}) -- inherited constraint"
        elif cand_arm == "fallback":
            verdict = f"ok: inherited mechanism fallback -- {c_detail.get('agr1_error')}"
        else:
            if not c_detail["finite"] or c_detail.get("agr1_attempted") != 1:
                verdict = (
                    f"FAIL: finite={c_detail['finite']} "
                    f"agr1_attempted={c_detail.get('agr1_attempted')}"
                )
                ok = False
            else:
                verdict = (
                    f"ok: finite, branch reached, accepted={c_detail['agr1_accepted']}, "
                    f"inv_err={c_detail['inverse_error']:.2e}"
                )
        print(f"{label:<34}{r_out:<20}{str(ref_arm):<26}{str(cand_arm):<26}{verdict}")

    print("=" * 100)
    print(f"OVERALL: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
