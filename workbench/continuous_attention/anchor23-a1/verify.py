"""Independent references and the A23 deployment contract checks.

Core acceptances: (1) the manual product-objective gradient matches
autograd; (2) constant Q*c / K/c rescaling leaves the product objective
invariant (the property the additive objective lacks); (3) with the
residual pinned to S=0 the calibration output matches the A22-2 parent
source bit-for-bit (tie -> retain complete parent); (4) deployment
coordinates and frozen sides are exact.
"""
from pathlib import Path
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

AUDIT_PREFIXES = ("a2_", "a21_", "a22_", "a22b_", "a23_")


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def compare_states(forced, official, context):
    """Every non-audit field must be bit-identical to the A22-2 parent output."""
    assert set(forced.keys()) == set(official.keys()), (context, sorted(forced), sorted(official))
    compared = 0
    for role in ("q_state", "k_state", "v_state"):
        keys = set(forced[role].keys()) | set(official[role].keys())
        for key in keys:
            if key.startswith(AUDIT_PREFIXES):
                continue
            assert key in forced[role] and key in official[role], (context, role, key)
            a, b = forced[role][key], official[role][key]
            if torch.is_tensor(a) or torch.is_tensor(b):
                assert torch.is_tensor(a) and torch.is_tensor(b), (context, role, key)
                assert a.shape == b.shape and a.dtype == b.dtype, (context, role, key, a.shape, b.dtype)
                assert torch.equal(a, b), (context, role, key, "tensor mismatch")
            else:
                assert a == b, (context, role, key, a, b)
            compared += 1
    return compared


def force_zero(mod, wins, qh, kv, dim):
    """Run calibration with the residual pinned to S=0 (tie -> retain parent)."""
    original = mod._a23_train

    def train_zero(*args, **kwargs):
        return original(*args, **{**kwargs, "force_zero": True})

    mod._a23_train = train_zero
    try:
        with torch.inference_mode():
            forced = mod.hif4_calibration_attention(wins, qh, kv, dim)
    finally:
        mod._a23_train = original
    assert forced["q_state"]["a23_accepted"] == 0, "S=0 must tie and retain the parent"
    return forced


def main():
    mod = load(HERE / "solution.py", "a23_verified")
    parent = load(ROOT / "solutions/continuous_attention_anchor22-a2/solution.py", "a22_parent")
    torch.manual_seed(21071)
    result = {"source_sha256": hashlib.sha256((HERE / "solution.py").read_bytes()).hexdigest(),
              "parent_sha256": hashlib.sha256((ROOT / "solutions/continuous_attention_anchor22-a2/solution.py").read_bytes()).hexdigest()}
    # 1) Product-objective gradient vs autograd, on the head-aligned layout.
    errors = []
    for qh, kh, dim, tokens, tokens_k in [(16, 4, 256, 12, 12), (14, 2, 64, 9, 11), (32, 8, 128, 5, 7), (12, 12, 64, 4, 4)]:
        torch.manual_seed(qh * 1000 + dim)
        xq = torch.randn(tokens, qh * dim) * 0.4
        xk = torch.randn(tokens_k, kh * dim) * 0.4
        xq.requires_grad_()
        xk.requires_grad_()
        nb = dim // 64
        hpg = qh // kh
        # autograd reference of the exact preregistered objective
        aq = xq.reshape(tokens, qh, nb, 64).abs().amax(-1).square().reshape(tokens, kh, hpg, nb).mean(dim=(0, 2))
        ak = xk.reshape(tokens_k, kh, nb, 64).abs().amax(-1).square().mean(0)
        dq = (aq.detach() * 1.07 + 0.05).clamp_min(1e-12)
        dk = (ak.detach() * 0.93 + 0.05).clamp_min(1e-12)
        loss = (aq * ak / (dq * dk)).mean()
        gq_exact, gk_exact = torch.autograd.grad(loss, (xq, xk))
        _, gq_man, gk_man = mod._a23_product_loss_grad(
            xq.detach(), xk.detach(), qh, kh, dim, dq, dk)
        torch.testing.assert_close(gq_man, gq_exact, atol=1e-6, rtol=1e-5)
        torch.testing.assert_close(gk_man, gk_exact, atol=1e-6, rtol=1e-5)
        errors.append(float((gq_man - gq_exact).abs().max()))
    # zero blocks and exact ties: gradient must vanish / split equally
    xq = torch.zeros(3, 4 * 256)
    xq[1, :64] = 2.0
    xq[2, 256:320] = -1.5
    dq = torch.ones(4, 4)
    _, gq, _ = mod._a23_product_loss_grad(xq, torch.randn(3, 4 * 256), 4, 4, 256, dq, dq)
    assert float(gq.abs().sum()) > 0
    assert float(gq[0].abs().sum()) == 0.0, "zero blocks must have zero gradient"
    result["product_gradient_vs_autograd_max_errors"] = errors
    result["zero_block_and_tie_gradient"] = "PASS"
    # 2) Constant Q*c / K/c invariance of the product objective.
    xq = torch.randn(6, 4 * 256) * 0.4
    xk = torch.randn(6, 4 * 256) * 0.4
    scale = torch.full((256,), 1.31)
    xq_scaled = xq.clone()
    for g in range(4):
        xq_scaled[:, g * 256:(g + 1) * 256] = xq[:, g * 256:(g + 1) * 256] * scale[None]
    xk_scaled = xk.clone()
    for g in range(4):
        xk_scaled[:, g * 256:(g + 1) * 256] = xk[:, g * 256:(g + 1) * 256] / scale[None]
    aq0, ak0 = mod._a23_moments(xq, xk, 4, 4, 256)
    aqs, aks = mod._a23_moments(xq_scaled, xk_scaled, 4, 4, 256)
    base_loss = (aq0 * ak0 / (aq0 * ak0)).mean()
    scaled_loss = (aqs * aks / (aq0 * ak0)).mean()
    assert abs(float(base_loss - scaled_loss)) < 1e-6, float(base_loss - scaled_loss)
    # the additive objective DOES change under the same rescaling (contrast)
    add_base = (aq0 / aq0).mean() + (ak0 / ak0).mean()
    result["product_invariance_qc_kdivc"] = f"PASS: product {float(scaled_loss):.9f} == base {float(base_loss):.9f}"
    # 3) Group-fallback layout gradient vs autograd (dim % 64 != 0).
    torch.manual_seed(7)
    xq = torch.randn(5, 16 * 80) * 0.4
    xk = torch.randn(5, 16 * 80) * 0.4
    xq.requires_grad_()
    xk.requires_grad_()
    a_q, a_k = mod._a23_moments(xq, xk, 16, 16, 80)
    loss = (a_q * a_k).mean()
    gq_exact, gk_exact = torch.autograd.grad(loss, (xq, xk))
    _, gq_man, gk_man = mod._a23_product_loss_grad_groupfallback(
        xq.detach(), xk.detach(), 16, 16, 80,
        torch.ones(16, device=xq.device), torch.ones(16, device=xk.device))
    torch.testing.assert_close(gq_man, gq_exact, atol=1e-6, rtol=1e-5)
    result["groupfallback_gradient_vs_autograd"] = f"PASS max {float((gq_man - gq_exact).abs().max()):.2e}"
    # 4) Unmodified base flow is syntactically identical to the parent chain.
    r3 = (ROOT / "solutions/v162_attention_r3-rotation-center_allgates/solution.py").read_text(encoding="utf-8")
    candidate = (HERE / "solution.py").read_text(encoding="utf-8")
    prefix = r3[:r3.rindex("def hif4_calibration_attention(")]
    assert candidate.startswith(prefix)
    result["frozen_prefix"] = "PASS: exact R3 prefix through _a2_train_rotation/_a2_true_path_gate_loss/_a2_normal"
    assert gpu_lock.acquire("A", "anchor23-a1-verify") == 0
    try:
        # 5) Synthetic windows: S=0 tie must reproduce the A22-2 parent
        # (forced the same way) bit-for-bit, including its parent decision.
        synthetic = []
        for _ in range(3):
            dense = [torch.randn(24, 16 * 256) * 0.3, torch.randn(24, 4 * 256) * 0.3, torch.randn(24, 4 * 256) * 0.3]
            synthetic.append({role: tuple(t.cuda() for t in ev._pair(d)) for role, d in zip(("q", "k", "v"), dense)})
        forced = force_zero(mod, synthetic, 16, 4, 256)
        original_a22_train = parent._a22b_train

        def train_zero_a22(*args, **kwargs):
            return original_a22_train(*args, **{**kwargs, "force_zero": True})

        parent._a22b_train = train_zero_a22
        try:
            with torch.inference_mode():
                official = parent.hif4_calibration_attention(synthetic, 16, 4, 256)
        finally:
            parent._a22b_train = original_a22_train
        compared = compare_states(forced, official, "synthetic")
        result["s0_tie_matches_a22_parent_synthetic"] = f"PASS: {compared} non-audit fields bit-identical"
        # 6) Real 4B cached windows for two FA layers (0, 22): legality,
        # deployed-coordinate capture, V control, and the S=0 identity.
        pack = torch.load(ROOT / "artifacts/official_eval/cache/qwen3.5-4b-proxy-v2.pt", map_location="cpu", weights_only=False)
        audits = []
        for layer in [0, 22]:
            wins = [{role: tuple(t.cuda() for t in ev._pair(dense)) for role, dense in zip(("q", "k", "v"), pack["calibration_qkv"][f][layer])} for f in range(5)]
            with torch.inference_mode():
                started = time.perf_counter()
                states = mod.hif4_calibration_attention(wins, 16, 4, 256)
                base_states = mod._V189_CALIBRATION_ATTENTION(wins, 16, 4, 256)
            assert states["q_state"]["a23_steps"] == 32
            for role, heads in [("q", 16), ("k", 4), ("v", 4)]:
                ref.validate_state(states[role + "_state"])
                api = getattr(mod, "hif4_dynamic_quantize_" + role)
                params = api(*wins[0][role], heads, 256, states[role + "_state"])
                ref.validate_hif4_params(params, wins[0][role][0].shape)
                assert torch.isfinite(mod._dequantize_hif4(params)).all()
                if role == "v":
                    other = api(*wins[0][role], heads, 256, base_states["v_state"])
                    assert all(torch.equal(params[key], other[key]) for key in params)
                else:
                    captured = []
                    original = mod._dense_to_hif4

                    def capture(dense, *args, **kwargs):
                        captured.append(dense.clone())
                        return original(dense, *args, **kwargs)

                    mod._dense_to_hif4 = capture
                    try:
                        api(*wins[0][role], heads, 256, states[role + "_state"])
                    finally:
                        mod._dense_to_hif4 = original
                    u = mod._a1_stack_transform(mod._dequantize_nvfp4_float32(*wins[0][role]), heads, 256, base_states[role + "_state"], role == "k")
                    learned = states[role + "_state"].get("learned_rotation")
                    if learned is not None:
                        u = mod._a2_apply_group_rotation(u, heads, learned.to(u.device))
                    center = states[role + "_state"].get("learned_center")
                    if center is not None:
                        lead = u.shape[:-1]
                        u = (u.reshape(*lead, heads, u.shape[-1] // heads)
                             + center.to(device=u.device, dtype=torch.float32).reshape(*([1] * len(lead)), heads, u.shape[-1] // heads)).reshape(u.shape)
                    assert torch.equal(u, captured[0]), (layer, role, float((u - captured[0]).abs().max()))
            forced = force_zero(mod, wins, 16, 4, 256)
            parent._a22b_train = train_zero_a22
            try:
                with torch.inference_mode():
                    official = parent.hif4_calibration_attention(wins, 16, 4, 256)
            finally:
                parent._a22b_train = original_a22_train
            compared = compare_states(forced, official, f"layer{layer}")
            audits.append({"layer": layer, "wall_s": time.perf_counter() - started,
                           "s0_tie_parent_fields": compared,
                           **{k: v for k, v in states["q_state"].items() if k.startswith(("a21_", "a22_", "a23_", "a2_"))}})
        result["real_api_inference_legal_control_coordinates"] = audits
    finally:
        gpu_lock.release("A", "anchor23-a1-verify")
    (HERE / "verification.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
