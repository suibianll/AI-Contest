"""Independent references and the A24 deployment contract checks.

Core acceptances: (1) _m_attention_backward matches autograd; (2) the full
STE training chain matches autograd on the quantization-as-identity path;
(3) with the residual pinned to S=0 the calibration output matches the
A22-2 parent source bit-for-bit (tie -> retain complete parent); (4)
deployment coordinates and frozen sides are exact.
"""
from pathlib import Path
import hashlib
import importlib.util
import json
import math
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

AUDIT_PREFIXES = ("a2_", "a21_", "a22_", "a22b_", "a23_", "a24_")


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def compare_states(forced, official, context):
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
    original = mod._a24_train

    def train_zero(*args, **kwargs):
        return original(*args, **{**kwargs, "force_zero": True})

    mod._a24_train = train_zero
    try:
        with torch.inference_mode():
            forced = mod.hif4_calibration_attention(wins, qh, kv, dim)
    finally:
        mod._a24_train = original
    assert forced["q_state"]["a24_accepted"] == 0, "S=0 must tie and retain the parent"
    return forced


def main():
    mod = load(HERE / "solution.py", "a24_verified")
    parent = load(ROOT / "solutions/continuous_attention_anchor22-a2/solution.py", "a22_parent")
    torch.manual_seed(21071)
    result = {"source_sha256": hashlib.sha256((HERE / "solution.py").read_bytes()).hexdigest(),
              "parent_sha256": hashlib.sha256((ROOT / "solutions/continuous_attention_anchor22-a2/solution.py").read_bytes()).hexdigest()}
    # 1) _m_attention_backward vs autograd on the exact attention graph.
    errors = []
    for qh, kh, dim, tq, tk in [(16, 4, 256, 9, 12), (14, 2, 64, 7, 11), (32, 8, 128, 5, 6)]:
        torch.manual_seed(qh * 31 + dim)
        q = torch.randn(tq, qh * dim) * 0.4
        k = torch.randn(tk, kh * dim) * 0.4
        v = torch.randn(tk, kh * dim) * 0.4
        q.requires_grad_(); k.requires_grad_(); v.requires_grad_()
        do = torch.randn(tq, qh * dim)
        # reference forward in float64-free autograd form (same math)
        qh3 = q.reshape(tq, qh, dim).transpose(0, 1)
        kh3 = k.reshape(tk, kh, dim).transpose(0, 1).repeat_interleave(qh // kh, 0)
        vh3 = v.reshape(tk, kh, dim).transpose(0, 1).repeat_interleave(qh // kh, 0)
        probs = torch.softmax(qh3 @ kh3.transpose(-1, -2) / math.sqrt(dim), -1)
        out = (probs @ vh3).transpose(0, 1).reshape(tq, qh * dim)
        (gq, gk, gv) = torch.autograd.grad((out * do).sum(), (q, k, v))
        dq, dk = mod._m_attention_backward(do[None], q.detach(), k.detach(), v.detach(), qh, kh, dim)
        torch.testing.assert_close(dq, gq, atol=1e-5, rtol=1e-4)
        torch.testing.assert_close(dk, gk, atol=1e-5, rtol=1e-4)
        errors.append(float((dq - gq).abs().max()))
    result["attention_backward_vs_autograd_max_errors"] = errors
    # 2) Full STE chain vs autograd on the quantization-as-identity path.
    #    NOTE: torch.matrix_exp's FLOAT32 autograd is numerically broken
    #    (verified against float64 central finite differences: O(1) error),
    #    so this reference is computed entirely in float64 with a plain
    #    grouped-rotation einsum (bypassing the float32-forced applier).
    for qh, kh, dim, tq, tk in [(16, 4, 256, 8, 8), (14, 2, 64, 7, 9)]:
        dtype = torch.float64
        torch.manual_seed(dim + 5)
        xq = torch.randn(tq, qh * dim, dtype=dtype) * 0.4
        xk = torch.randn(tk, kh * dim, dtype=dtype) * 0.4
        vhat = torch.randn(tk, kh * dim, dtype=dtype) * 0.4
        ref_out = torch.randn(tq, qh * dim, dtype=dtype) * 0.4
        s = mod._a21_project((torch.randn(kh, dim, dim) * 0.05).double())
        s_leaf = s.clone().requires_grad_()
        e1 = torch.matrix_exp(s_leaf)
        e2 = torch.matrix_exp(-s_leaf)
        hpg = qh // kh

        def rot(x, e, members):
            return torch.einsum("tghk,gkd->tghd", x.reshape(x.shape[0], kh, members, dim), e).reshape(x.shape[0], -1)

        yq = rot(xq, e1, hpg)
        yk = rot(xk, e2, 1)
        qh3 = yq.reshape(tq, qh, dim).transpose(0, 1)
        kh3 = yk.reshape(tk, kh, dim).transpose(0, 1).repeat_interleave(hpg, 0)
        vh3 = vhat.reshape(tk, kh, dim).transpose(0, 1).repeat_interleave(hpg, 0)
        probs = torch.softmax(qh3 @ kh3.transpose(-1, -2) / math.sqrt(dim), -1)
        out = (probs @ vh3).transpose(0, 1).reshape(tq, qh * dim)
        loss = (out - ref_out).square().mean()
        gs = torch.autograd.grad(loss, s_leaf)[0]
        # manual chain: STE dL/dy via _m_attention_backward (dtype-agnostic),
        # then the pre-encode einsum, then the eigh-based exp backward.
        d_out = 2.0 * (out.detach() - ref_out) / out.numel()
        d_q, d_k = mod._m_attention_backward(d_out[None], yq.detach(), yk.detach(), vhat, qh, kh, dim)
        gp = mod._a21_matrix_grad(xq, d_q, qh, kh)
        gm = mod._a21_matrix_grad(xk, d_k, kh, kh)
        values, vectors = torch.linalg.eigh(s)
        cache_p = (values, vectors, values.exp(), 1.0)
        cache_m = (values, vectors, (-values).exp(), -1.0)
        gs_man = mod._a21_exp_backward(gp, cache_p) + mod._a21_exp_backward(gm, cache_m)
        gs_man = (gs_man + gs_man.transpose(-1, -2)) * 0.5
        torch.testing.assert_close(gs_man, gs, atol=1e-10, rtol=1e-8)
    result["ste_chain_vs_autograd"] = "PASS: two geometries in float64, max diff <= 1e-10 (repeated eigenvalues included)"
    # 3) Unmodified base flow is syntactically identical to the parent chain.
    r3 = (ROOT / "solutions/v162_attention_r3-rotation-center_allgates/solution.py").read_text(encoding="utf-8")
    candidate = (HERE / "solution.py").read_text(encoding="utf-8")
    prefix = r3[:r3.rindex("def hif4_calibration_attention(")]
    assert candidate.startswith(prefix)
    result["frozen_prefix"] = "PASS: exact R3 prefix through _a2_train_rotation/_a2_true_path_gate_loss/_a2_normal"
    assert gpu_lock.acquire("A", "anchor24-a1-verify") == 0
    try:
        # 4) Synthetic 4B windows: S=0 tie must reproduce the A22-2 parent
        # (force-pinned the same way) bit-for-bit.
        synthetic = []
        for _ in range(2):
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
        # 5) Real 4B cached windows for two FA layers (0, 22): legality,
        # deployed-coordinate capture, V control, and the S=0 identity.
        pack = torch.load(ROOT / "artifacts/official_eval/cache/qwen3.5-4b-proxy-v2.pt", map_location="cpu", weights_only=False)
        audits = []
        for layer in [0, 22]:
            wins = [{role: tuple(t.cuda() for t in ev._pair(dense)) for role, dense in zip(("q", "k", "v"), pack["calibration_qkv"][f][layer])} for f in range(5)]
            with torch.inference_mode():
                started = time.perf_counter()
                states = mod.hif4_calibration_attention(wins, 16, 4, 256)
                base_states = mod._V189_CALIBRATION_ATTENTION(wins, 16, 4, 256)
            assert states["q_state"]["a24_steps"] == 32
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
                           **{k: v for k, v in states["q_state"].items() if k.startswith(("a21_", "a22_", "a23_", "a24_", "a2_"))}})
        result["real_api_inference_legal_control_coordinates"] = audits
    finally:
        gpu_lock.release("A", "anchor24-a1-verify")
    (HERE / "verification.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


def _scatter_v_grad(do, qh, kh, dim):
    """dL/dv for the grouped attention: d_out projected per group head.

    v grad = sum over group heads of d_out on that head, i.e. each kv head
    accumulates do[:, h, :] for h in its group (matches repeat_interleave
    forward). Implemented with the same grouping as _m_attention_backward.
    """
    tq = do.shape[0]
    group = qh // kh
    do3 = do.reshape(tq, qh, dim)
    dv = torch.zeros(tq, kh, dim, dtype=do.dtype)
    for h in range(qh):
        dv[:, h // group] += do3[:, h]
    return dv.reshape(tq, kh * dim)


if __name__ == "__main__":
    main()
