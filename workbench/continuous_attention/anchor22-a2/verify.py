"""Independent references and the A22-2 deployment contract checks.

Core acceptance: with the residual pinned to S=0 (training forced to zero),
the calibration output must match the official R3 source bit-for-bit on
synthetic windows and on real cached calibration windows, and the compiled
deployment pair (Rq@exp(S), Rk@exp(-S), c@exp(-S)) must reproduce the
training target ZK@exp(-S) including the additive K-center.
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

AUDIT_PREFIXES = ("a2_", "a21_", "a22_", "a22b_")


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def compare_states(forced, official, context):
    """Every non-audit field must be bit-identical to the official R3 output."""
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
    original = mod._a22b_train

    def train_zero(*args, **kwargs):
        return original(*args, **{**kwargs, "force_zero": True})

    mod._a22b_train = train_zero
    try:
        with torch.inference_mode():
            forced = mod.hif4_calibration_attention(wins, qh, kv, dim)
    finally:
        mod._a22b_train = original
    assert forced["q_state"]["a22_accepted"] == 0, "S=0 must tie and retain the parent"
    return forced


def main():
    mod = load(HERE / "solution.py", "a22b_verified")
    r3 = load(ROOT / "solutions/v162_attention_r3-rotation-center_allgates/solution.py", "r3_official")
    torch.manual_seed(21071)
    result = {"source_sha256": hashlib.sha256((HERE / "solution.py").read_bytes()).hexdigest(),
              "parent_sha256": hashlib.sha256((ROOT / "solutions/v162_attention_r3-rotation-center_allgates/solution.py").read_bytes()).hexdigest()}
    # Deployment compile identity: UK@(Rk@em) + (c@em) == (UK@Rk + c)@em.
    # Floating point is not exactly distributive, so assert closeness, and
    # assert the S=0 case is EXACT (tq=rq, tk=rk, c_new=c).
    torch.manual_seed(7)
    kh, dim, tokens = 3, 64, 17
    q, _ = torch.linalg.qr(torch.randn(dim, dim))
    rq = q.expand(kh, dim, dim).contiguous()
    rk = rq.clone()
    c = torch.randn(kh, dim) * 0.05
    uq = torch.randn(tokens, kh * dim)
    uk = torch.randn(tokens, kh * dim)
    s = mod._a21_project(torch.randn(kh, dim, dim) * 0.02)
    ep, _ = mod._a21_exp(s)
    em, _ = mod._a21_exp(s, -1)
    tq, tk, c_new = rq @ ep, rk @ em, (c.unsqueeze(-2) @ em).squeeze(-2)
    deployed = mod._a2_apply_group_rotation(uk, kh, tk)
    deployed = (deployed.reshape(tokens, kh, dim) + c_new.unsqueeze(0)).reshape(tokens, kh * dim)
    # Training target: the S-transformed parent coordinate ZK@em, per group.
    zk = mod._a2_apply_group_rotation(uk, kh, rk).reshape(tokens, kh, dim) + c.unsqueeze(0)
    target = torch.einsum("tgk,gkd->tgd", zk, em).reshape(tokens, kh * dim)
    assert float((deployed - target).abs().max()) < 5e-6, float((deployed - target).abs().max())
    eye = torch.eye(dim).expand(kh, dim, dim)
    tq0, tk0, c0 = rq @ eye, rk @ eye, (c.unsqueeze(-2) @ eye).squeeze(-2)
    assert torch.equal(tq0, rq) and torch.equal(tk0, rk) and torch.equal(c0, c)
    result["deployment_compile_identity"] = "PASS: (UK@Rk + c)@em == UK@(Rk@em) + c@em; S=0 exact"
    # No shared grouped applier in the reference. Unique matrices expose GQA
    # broadcasting/axis mistakes that shared-code replica tests cannot detect.
    for qh_, kh_, dim_ in [(14, 2, 64), (4, 2, 32), (16, 16, 80), (8, 2, 128)]:
        s = torch.randn(kh_, dim_, dim_) * 0.006
        s = mod._a21_project(s)
        ep, _ = mod._a21_exp(s)
        em, _ = mod._a21_exp(s, -1)
        q, k = torch.randn(7, qh_, dim_), torch.randn(11, kh_, dim_)
        actual = mod._a2_apply_group_rotation(q.flatten(1), qh_, ep).reshape_as(q)
        independent = torch.stack([q[:, h] @ ep[h // (qh_ // kh_)] for h in range(qh_)], 1)
        torch.testing.assert_close(actual, independent, atol=2e-6, rtol=2e-6)
        kt = mod._a2_apply_group_rotation(k.flatten(1), kh_, em).reshape_as(k)
        for h in range(qh_):
            g = h // (qh_ // kh_)
            torch.testing.assert_close(actual[:, h] @ kt[:, g].T, q[:, h] @ k[:, g].T, atol=1e-4, rtol=3e-5)
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
    # Unmodified sides are syntactically identical: the candidate keeps the
    # whole R3 source up to its final calibration wrapper.
    parent = (ROOT / "solutions/v162_attention_r3-rotation-center_allgates/solution.py").read_text(encoding="utf-8")
    candidate = (HERE / "solution.py").read_text(encoding="utf-8")
    prefix = parent[:parent.rindex("def hif4_calibration_attention(")]
    assert candidate.startswith(prefix)
    result["frozen_prefix"] = "PASS: exact R3 prefix through _a2_train_rotation/_a2_true_path_gate_loss/_a2_normal"
    assert gpu_lock.acquire("A", "anchor22-a2-verify") == 0
    try:
        # Synthetic windows: S=0 tie must reproduce official R3 exactly.
        synthetic = []
        for _ in range(3):
            dense = [torch.randn(96, 14 * 64) * 0.3, torch.randn(96, 2 * 64) * 0.3, torch.randn(96, 2 * 64) * 0.3]
            synthetic.append({role: tuple(t.cuda() for t in ev._pair(d)) for role, d in zip(("q", "k", "v"), dense)})
        forced = force_zero(mod, synthetic, 14, 2, 64)
        with torch.inference_mode():
            official = r3.hif4_calibration_attention(synthetic, 14, 2, 64)
        compared = compare_states(forced, official, "synthetic")
        result["s0_tie_matches_r3_synthetic"] = f"PASS: {compared} non-audit fields bit-identical"
        # Real cached windows for L0/L23: default run legality, deployed
        # coordinate capture (candidate or parent), V control, S=0 identity.
        pack = torch.load(ROOT / "artifacts/official_eval/cache/qwen2.5-0.5b-proxy-v2.pt", map_location="cpu", weights_only=False)
        audits = []
        for layer in [0, 23]:
            wins = [{role: tuple(t.cuda() for t in ev._pair(dense)) for role, dense in zip(("q", "k", "v"), pack["calibration_qkv"][f][layer])} for f in range(5)]
            with torch.inference_mode():
                started = time.perf_counter()
                states = mod.hif4_calibration_attention(wins, 14, 2, 64)
                base = mod._V189_CALIBRATION_ATTENTION(wins, 14, 2, 64)
            assert states["q_state"]["a22b_steps"] == 32
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
                    captured = []
                    original = mod._dense_to_hif4

                    def capture(dense, *args, **kwargs):
                        captured.append(dense.clone())
                        return original(dense, *args, **kwargs)

                    mod._dense_to_hif4 = capture
                    try:
                        api(*wins[0][role], heads, 64, states[role + "_state"])
                    finally:
                        mod._dense_to_hif4 = original
                    u = mod._a1_stack_transform(mod._dequantize_nvfp4_float32(*wins[0][role]), heads, 64, base[role + "_state"], role == "k")
                    learned = states[role + "_state"].get("learned_rotation")
                    if learned is not None:
                        u = mod._a2_apply_group_rotation(u, heads, learned.to(u.device))
                    center = states[role + "_state"].get("learned_center")
                    if center is not None:
                        lead = u.shape[:-1]
                        u = (u.reshape(*lead, heads, u.shape[-1] // heads)
                             + center.to(device=u.device, dtype=torch.float32).reshape(*([1] * len(lead)), heads, u.shape[-1] // heads)).reshape(u.shape)
                    assert torch.equal(u, captured[0]), (layer, role, float((u - captured[0]).abs().max()))
            forced = force_zero(mod, wins, 14, 2, 64)
            with torch.inference_mode():
                official = r3.hif4_calibration_attention(wins, 14, 2, 64)
            compared = compare_states(forced, official, f"layer{layer}")
            audits.append({"layer": layer, "wall_s": time.perf_counter() - started,
                           "s0_tie_r3_fields": compared,
                           **{k: v for k, v in states["q_state"].items() if k.startswith(("a21_", "a22_", "a2_"))}})
        result["real_api_inference_legal_control_coordinates"] = audits
    finally:
        gpu_lock.release("A", "anchor22-a2-verify")
    (HERE / "verification.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
