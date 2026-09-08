"""Independent reachability, parent-fallback, and deployment checks."""

from pathlib import Path
import hashlib
import importlib.util
import json

import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
import sys
sys.path.insert(0, str(ROOT / "evaluator"))
import reference_hif4 as ref


def load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_windows(device: torch.device, *, q_heads: int = 4, kv_heads: int = 2,
                 head_dim: int = 64, tokens: int = 32):
    windows = []
    for index in range(5):
        generator = torch.Generator(device=device).manual_seed(21071 + index)
        q = torch.randn(tokens, q_heads * head_dim, device=device, generator=generator)
        k = torch.randn(tokens, kv_heads * head_dim, device=device, generator=generator)
        v = torch.randn(tokens, kv_heads * head_dim, device=device, generator=generator)
        q_scale = torch.ones(tokens, q_heads * head_dim // 16, device=device)
        k_scale = torch.ones(tokens, kv_heads * head_dim // 16, device=device)
        v_scale = torch.ones(tokens, kv_heads * head_dim // 16, device=device)
        windows.append({"q": (q, q_scale), "k": (k, k_scale), "v": (v, v_scale)})
    return windows


def compare_parent_fields(candidate_state, parent_state):
    ignored = {"diag_arm", "diag_attempted", "diag_accepted", "diag_fit_windows",
               "diag_gate_windows", "fit_windows", "a_mean", "b_mean", "d_norm",
               "d_max_abs", "d_mean_abs", "d_zero_fraction", "fold_mode",
               "diag_gate", "diag_q_changed_codes", "diag_k_changed_codes"}
    assert set(candidate_state) == set(parent_state)
    for role in ("q_state", "k_state", "v_state"):
        keys = (set(candidate_state[role]) | set(parent_state[role])) - ignored
        for key in keys:
            assert key in candidate_state[role] and key in parent_state[role], (role, key)
            left, right = candidate_state[role][key], parent_state[role][key]
            if torch.is_tensor(left) or torch.is_tensor(right):
                assert torch.is_tensor(left) and torch.is_tensor(right)
                assert left.dtype == right.dtype and left.shape == right.shape
                assert torch.equal(left, right), (role, key)
            else:
                assert left == right, (role, key, left, right)


def main():
    candidate = load(HERE / "candidate" / "solution.py", "diag_verify_candidate")
    parent = load(ROOT / "solution.py", "diag_verify_parent")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    result = {
        "candidate_sha256": hashlib.sha256(
            (HERE / "candidate" / "solution.py").read_bytes()
        ).hexdigest(),
        "parent_sha256": hashlib.sha256((ROOT / "solution.py").read_bytes()).hexdigest(),
    }

    # Verify both compilation modes and the exact reciprocal relationship in
    # the final parent coordinates, including an additive learned K center.
    kv, hd, qh = 2, 64, 4
    eye = torch.eye(hd).expand(kv, hd, hd).clone()
    center = torch.randn(kv, hd) * 0.05
    parent_states = {
        "q_state": {"learned_rotation": eye.clone()},
        "k_state": {"learned_rotation": eye.clone(), "learned_center": center.clone()},
        "v_state": {},
    }
    a = torch.ones(kv, hd)
    b = torch.ones(kv, hd)
    b[:, 0] = 4.0
    q_compiled, k_compiled, info = candidate._attn_diag_reciprocal_balance(
        parent_states, a, b, qh, kv, hd
    )
    assert info["fold_mode"] == "learned_rotation_and_center"
    q_raw = torch.randn(7, qh * hd)
    k_raw = torch.randn(7, kv * hd)
    q_parent = candidate._attn_diag_parent_dense(
        q_raw, parent_states["q_state"], qh, hd, is_k=False
    )
    k_parent = candidate._attn_diag_parent_dense(
        k_raw, parent_states["k_state"], kv, hd, is_k=True
    )
    d = info["d"]
    q_expected = q_parent * d.exp().repeat_interleave(qh // kv, dim=0).reshape(1, -1)
    k_expected = k_parent * d.neg().exp().reshape(1, -1)
    q_actual = candidate._attn_diag_parent_dense(q_raw, q_compiled, qh, hd, is_k=False)
    k_actual = candidate._attn_diag_parent_dense(k_raw, k_compiled, kv, hd, is_k=True)
    torch.testing.assert_close(q_actual, q_expected, atol=3e-5, rtol=3e-5)
    torch.testing.assert_close(k_actual, k_expected, atol=3e-5, rtol=3e-5)
    result["folded_rotation_center_compile"] = "PASS"

    no_rotation_states = {
        "q_state": {}, "k_state": {"center_mode": 0}, "v_state": {}
    }
    q_compiled, k_compiled, info = candidate._attn_diag_reciprocal_balance(
        no_rotation_states, a, b, qh, kv, hd
    )
    assert info["fold_mode"] == "post_transform_vector"
    assert "diag_scale" in q_compiled and "diag_scale" in k_compiled
    q_parent = candidate._attn_diag_parent_dense(
        q_raw, no_rotation_states["q_state"], qh, hd, is_k=False
    )
    k_parent = candidate._attn_diag_parent_dense(
        k_raw, no_rotation_states["k_state"], kv, hd, is_k=True
    )
    q_actual = candidate._attn_diag_parent_dense(q_raw, q_compiled, qh, hd, is_k=False)
    k_actual = candidate._attn_diag_parent_dense(k_raw, k_compiled, kv, hd, is_k=True)
    torch.testing.assert_close(
        q_actual, q_parent * q_compiled["diag_scale"].reshape(1, -1), atol=3e-5, rtol=3e-5
    )
    torch.testing.assert_close(
        k_actual, k_parent * k_compiled["diag_scale"].reshape(1, -1), atol=3e-5, rtol=3e-5
    )
    result["post_transform_vector_compile"] = "PASS"

    # Force a zero proposal.  It must reject on a strict tie and preserve all
    # functional parent fields, proving the fallback is reachable.
    windows = make_windows(device)
    original_energy = candidate._attn_diag_error_energy
    candidate._attn_diag_error_energy = lambda *args, **kwargs: (
        torch.ones(kv, hd, device=device),
        torch.ones(kv, hd, device=device),
        {"fit_windows": 3, "a_mean": 1.0, "b_mean": 1.0},
    )
    try:
        with torch.inference_mode():
            no_op = candidate.hif4_calibration_attention(windows, qh, kv, hd)
            parent_result = parent.hif4_calibration_attention(windows, qh, kv, hd)
    finally:
        candidate._attn_diag_error_energy = original_energy
    assert no_op["q_state"]["diag_attempted"] == 1
    assert no_op["q_state"]["diag_accepted"] == 0
    assert no_op["q_state"]["diag_arm"] == "parent"
    compare_parent_fields(no_op, parent_result)
    for role, heads in (("q", qh), ("k", kv), ("v", kv)):
        state = no_op[role + "_state"]
        ref.validate_state(state)
        params = getattr(candidate, "hif4_dynamic_quantize_" + role)(
            *windows[0][role], heads, hd, state
        )
        ref.validate_hif4_params(params, windows[0][role][0].shape)
        assert bool(torch.isfinite(candidate._dequantize_hif4(params)).all())
    result["zero_proposal_parent_fallback"] = "PASS"

    # Force one deterministic nonzero proposal and the normal two-window
    # scorer to pass, proving the algorithm is actually reachable rather than
    # a permanent no-op.
    original_energy = candidate._attn_diag_error_energy
    original_gate = candidate._attn_diag_gate_passes
    candidate._attn_diag_error_energy = lambda *args, **kwargs: (
        torch.ones(kv, hd, device=device),
        torch.cat((torch.full((kv, 1), 4.0, device=device), torch.ones(kv, hd - 1, device=device)), dim=1),
        {"fit_windows": 3, "a_mean": 1.0, "b_mean": 1.046875},
    )
    candidate._attn_diag_gate_passes = lambda parent_loss, candidate_loss: True
    try:
        with torch.inference_mode():
            reachable = candidate.hif4_calibration_attention(windows, qh, kv, hd)
    finally:
        candidate._attn_diag_error_energy = original_energy
        candidate._attn_diag_gate_passes = original_gate
    assert reachable["q_state"]["diag_attempted"] == 1
    assert reachable["q_state"]["diag_accepted"] == 1
    assert reachable["q_state"]["diag_arm"] == "accepted"
    assert float(reachable["q_state"]["d_max_abs"]) > 0.0
    result["forced_reachability"] = "PASS"

    (HERE / "verification.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
