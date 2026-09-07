"""Independent gradient/GQA references and real deployment contract checks."""
from pathlib import Path
import ast
import hashlib
import importlib.util
import json
import sys
import time
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "evaluator"))
sys.path.insert(0, str(ROOT / "workbench/v162_attention"))
import official_eval as ev
import reference_hif4 as ref
import gpu_lock


def main():
    spec = importlib.util.spec_from_file_location("a21_verified", HERE / "solution.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    torch.manual_seed(21071)
    result = {"source_sha256": hashlib.sha256((HERE / "solution.py").read_bytes()).hexdigest()}
    # No shared grouped applier in the reference. Unique matrices expose GQA
    # broadcasting/axis mistakes that shared-code replica tests cannot detect.
    for qh, kh, dim in [(14, 2, 64), (4, 2, 32), (16, 16, 80), (8, 2, 128)]:
        s = torch.randn(kh, dim, dim) * 0.006
        s = mod._a21_project(s)
        ep, _ = mod._a21_exp(s)
        em, _ = mod._a21_exp(s, -1)
        q, k = torch.randn(7, qh, dim), torch.randn(11, kh, dim)
        actual = mod._a2_apply_group_rotation(q.flatten(1), qh, ep).reshape_as(q)
        independent = torch.stack([q[:, h] @ ep[h // (qh // kh)] for h in range(qh)], 1)
        torch.testing.assert_close(actual, independent, atol=2e-6, rtol=2e-6)
        kt = mod._a2_apply_group_rotation(k.flatten(1), kh, em).reshape_as(k)
        for h in range(qh):
            g = h // (qh // kh)
            torch.testing.assert_close(actual[:, h] @ kt[:, g].T, q[:, h] @ k[:, g].T, atol=1e-4, rtol=3e-5)
        eigen = torch.linalg.eigvalsh(s)
        assert float(eigen.abs().max()) <= 0.5 * __import__("math").log(2) + 2e-6
        assert float(eigen.sum(-1).abs().max()) < 1e-5
    result["independent_gqa_and_inverse"] = "PASS: four geometries, unequal Q/K lengths"
    # Matrix exponential Frechet derivative, including repeated eigenvalues.
    errors = []
    for zero in [True, False]:
        raw = torch.zeros(2, 8, 8) if zero else torch.randn(2, 8, 8) * 0.08
        raw.requires_grad_()
        s = (raw + raw.transpose(-1, -2)) * 0.5
        gp, gm = torch.randn_like(s), torch.randn_like(s)
        loss = (torch.matrix_exp(s) * gp).sum() + (torch.matrix_exp(-s) * gm).sum()
        exact = torch.autograd.grad(loss, raw)[0]
        _, cp = mod._a21_exp(s.detach())
        _, cm = mod._a21_exp(s.detach(), -1)
        manual = mod._a21_exp_backward(gp, cp) + mod._a21_exp_backward(gm, cm)
        errors.append(float((exact - manual).abs().max()))
        torch.testing.assert_close(exact, manual, atol=2e-5, rtol=2e-5)
    # Scale derivative covers ties, zero blocks and normal blocks.
    x = torch.randn(3, 128)
    x[0, :64] = 0
    x[1, :64] = 2
    x.requires_grad_()
    denominator = torch.ones(3, 2, 1)
    exact_loss = (x.reshape(3, 2, 64).abs().amax(-1, keepdim=True) / denominator).square().mean()
    exact = torch.autograd.grad(exact_loss, x)[0]
    _, manual = mod._a21_scale_loss_grad(x.detach(), denominator)
    torch.testing.assert_close(exact, manual)
    result["gradient_max_errors"] = errors
    result["amax_ties_zero_gradient"] = "PASS"
    # Unmodified sides are syntactically identical, including the full prefix
    # which contains their shared dependencies; actual V controls follow below.
    parent = (ROOT / "solutions/v162_attention_r3-rotation-center_allgates/solution.py").read_text(encoding="utf-8")
    candidate = (HERE / "solution.py").read_text(encoding="utf-8")
    prefix = parent[:parent.index("def _a2_train_rotation(")]
    assert candidate.startswith(prefix)
    result["frozen_prefix"] = "PASS: exact parent prefix, Linear APIs and V dependencies"
    assert gpu_lock.acquire("A", "anchor21-a1-verify") == 0
    try:
        pack = torch.load(ROOT / "artifacts/official_eval/cache/qwen2.5-0.5b-proxy-v2.pt", map_location="cpu", weights_only=False)
        audits = []
        for layer in [0, 23]:
            wins = [{role: tuple(t.cuda() for t in ev._pair(dense)) for role, dense in zip(("q", "k", "v"), pack["calibration_qkv"][f][layer])} for f in range(5)]
            with torch.inference_mode():
                started = time.perf_counter()
                states = mod.hif4_calibration_attention(wins, 14, 2, 64)
                base = mod._V189_CALIBRATION_ATTENTION(wins, 14, 2, 64)
                assert states["q_state"]["a21_steps"] == 32
                for role, heads in [("q", 14), ("k", 2), ("v", 2)]:
                    ref.validate_state(states[role + "_state"])
                    api = getattr(mod, "hif4_dynamic_quantize_" + role)
                    params = api(*wins[0][role], heads, 64, states[role + "_state"])
                    ref.validate_hif4_params(params, wins[0][role][0].shape)
                    assert torch.isfinite(mod._dequantize_hif4(params)).all()
                    if role == "v":
                        other = api(*wins[0][role], heads, 64, base["v_state"])
                        assert all(torch.equal(params[key], other[key]) for key in params)
                    else:
                        # Capture the real encoder input, independently of the
                        # training preprocessor, then compare coordinates.
                        captured = []
                        original = mod._dense_to_hif4
                        def capture(dense, *args, **kwargs):
                            captured.append(dense.clone())
                            return original(dense, *args, **kwargs)
                        mod._dense_to_hif4 = capture
                        try:
                            api(*wins[0][role], heads, 64, base[role + "_state"])
                        finally:
                            mod._dense_to_hif4 = original
                        u = mod._a1_stack_transform(mod._dequantize_nvfp4_float32(*wins[0][role]), heads, 64, base[role + "_state"], role == "k")
                        assert torch.equal(u, captured[0]), (layer, role, float((u - captured[0]).abs().max()))
                audits.append({"layer": layer, "wall_s": time.perf_counter() - started, **{k: v for k, v in states["q_state"].items() if k.startswith("a21_")}})
        result["real_api_inference_legal_control_coordinates"] = audits
    finally:
        gpu_lock.release("A", "anchor21-a1-verify")
    (HERE / "verification.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
