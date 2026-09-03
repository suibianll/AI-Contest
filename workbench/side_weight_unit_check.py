"""Unit checks for the official side-weight calibration candidates v162-v164.

1. Isolated import of all three files (no repository context).
2. Six APIs present.
3. Standard codec bit-exactness against evaluator/reference_hif4.py on NVFP4
   inputs (CPU and CUDA).
4. v163 Attention outputs == v162 Attention outputs; v164 Linear outputs ==
   v162 Linear outputs (same standard codec).
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "evaluator"))

import reference_hif4  # noqa: E402
from nvfp4_sim import nvfp4_encode  # noqa: E402

CANDIDATES = {
    "v162": ROOT / "solutions/20260903_v162_standard-baseline-both_scoreNA_timeNA/solution.py",
    "v163": ROOT / "solutions/20260903_v163_v160-linear_standard-attn_scoreNA_timeNA/solution.py",
    "v164": ROOT / "solutions/20260903_v164_standard-linear_v160-attn_scoreNA_timeNA/solution.py",
}
API_NAMES = [
    "hif4_calibration_and_quantize_weight",
    "hif4_dynamic_quantize_activation",
    "hif4_calibration_attention",
    "hif4_dynamic_quantize_q",
    "hif4_dynamic_quantize_k",
    "hif4_dynamic_quantize_v",
]


def load_isolated(path: Path):
    spec = importlib.util.spec_from_file_location(f"cand_{path.parent.name}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def params_equal(a, b) -> bool:
    if set(a) != set(b):
        return False
    for key in a:
        ta = a[key].to(torch.float32)
        tb = b[key].to(torch.float32)
        if ta.shape != tb.shape or not bool(torch.equal(ta, tb)):
            return False
    return True


def main() -> None:
    torch.manual_seed(0)
    modules = {}
    for name, path in CANDIDATES.items():
        module = load_isolated(path)
        for api in API_NAMES:
            assert hasattr(module, api), f"{name} missing {api}"
        print(f"[A] {name}: isolated import + 6 APIs OK ({path.parent.name})")
        modules[name] = module

    for device in ("cpu", "cuda"):
        if device == "cuda" and not torch.cuda.is_available():
            print("[A] CUDA unavailable, skipping CUDA checks")
            continue
        dev = torch.device(device)
        # Linear-shaped and attention-shaped NVFP4 inputs.
        weight = torch.randn(96, 896, device=dev) * 0.05
        activation = torch.randn(128, 896, device=dev) * 0.1
        qkv = torch.randn(64, 14 * 64, device=dev) * 0.08
        w_q, w_s = nvfp4_encode(weight)
        a_q, a_s = nvfp4_encode(activation)
        t_q, t_s = nvfp4_encode(qkv)
        w_q, w_s, a_q, a_s, t_q, t_s = (
            t.to(dev) for t in (w_q, w_s, a_q, a_s, t_q, t_s)
        )

        ref_w = reference_hif4.encode_standard_hif4(
            reference_hif4.dequantize_nvfp4(w_q, w_s).to(torch.float32)
        )
        ref_a = reference_hif4.encode_standard_hif4(
            reference_hif4.dequantize_nvfp4(a_q, a_s).to(torch.float32)
        )
        ref_t = reference_hif4.encode_standard_hif4(
            reference_hif4.dequantize_nvfp4(t_q, t_s).to(torch.float32)
        )

        for name, module in modules.items():
            if name in ("v162", "v164"):
                out_w = module.hif4_dynamic_quantize_activation(a_q, a_s, {})
                assert params_equal(out_w, ref_a), f"{name} activation != reference ({device})"
                out_w2 = module.hif4_calibration_and_quantize_weight(w_q, w_s, [(a_q, a_s)])
                assert params_equal(out_w2["weight_params"], ref_w), f"{name} weight != reference ({device})"
                reference_hif4.validate_hif4_params(
                    out_w2["weight_params"], weight.shape
                )
            if name in ("v162", "v163"):
                states = module.hif4_calibration_attention([{"q": (t_q, t_s), "k": (t_q, t_s), "v": (t_q, t_s)}], 14, 2, 64)
                assert set(states) == {"q_state", "k_state", "v_state"}, f"{name} attention state keys ({device})"
                reference_hif4.validate_state(states["q_state"])
                out_q = module.hif4_dynamic_quantize_q(t_q, t_s, 14, 64, states["q_state"])
                out_k = module.hif4_dynamic_quantize_k(t_q, t_s, 2, 64, states["k_state"])
                out_v = module.hif4_dynamic_quantize_v(t_q, t_s, 2, 64, states["v_state"])
                for tag, out in (("q", out_q), ("k", out_k), ("v", out_v)):
                    assert params_equal(out, ref_t), f"{name} {tag} != reference ({device})"
                    reference_hif4.validate_hif4_params(out, qkv.shape)
        print(f"[A] standard codec bit-exact vs reference on {device}: OK (v162 both, v163 attn, v164 linear)")

    # Cross-candidate identity: v163 attention == v162 attention; v164 linear == v162 linear.
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    a_q, a_s, t_q, t_s = (t.to(dev) for t in (a_q, a_s, t_q, t_s))
    s162 = modules["v162"].hif4_calibration_attention([{"q": (t_q, t_s), "k": (t_q, t_s), "v": (t_q, t_s)}], 14, 2, 64)
    s163 = modules["v163"].hif4_calibration_attention([{"q": (t_q, t_s), "k": (t_q, t_s), "v": (t_q, t_s)}], 14, 2, 64)
    for key in ("q_state", "k_state", "v_state"):
        q162 = modules["v162"].hif4_dynamic_quantize_q(t_q, t_s, 14, 64, s162[key])
        q163 = modules["v163"].hif4_dynamic_quantize_q(t_q, t_s, 14, 64, s163[key])
        assert params_equal(q162, q163), "v163 attention Q != v162 attention Q"
    out162 = modules["v162"].hif4_dynamic_quantize_activation(a_q, a_s, {})
    out164 = modules["v164"].hif4_dynamic_quantize_activation(a_q, a_s, {})
    assert params_equal(out162, out164), "v164 linear activation != v162"
    print("[A] cross-candidate identity: v163 attn == v162 attn, v164 linear == v162 linear")
    print("ALL UNIT CHECKS PASSED")


if __name__ == "__main__":
    main()
