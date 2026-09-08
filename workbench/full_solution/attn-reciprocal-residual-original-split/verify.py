"""Verify the A22-2 original-split reciprocal-residual candidate on synthetic data.

Checks: reciprocal compile (exp(S) @ exp(-S) == I), synchronized K-center
compile (c_new = c @ exp(-S)), real training branch executed and adopted by a
synthetic layer with attempted/accepted counters, parent fallback branch,
legal state/finite outputs, and the original 4-fit + 1-gate window split.
"""

from pathlib import Path
import hashlib
import importlib.util
import json
import sys

import torch


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "evaluator"))
import reference_hif4 as ref

torch.set_num_threads(1)


def load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_windows(device: torch.device, *, q_heads: int = 4, kv_heads: int = 2,
                 head_dim: int = 64, tokens: int = 8, q_spike: float = 0.0):
    windows = []
    for index in range(5):
        generator = torch.Generator(device=device).manual_seed(21071 + index)
        q = torch.randn(tokens, q_heads * head_dim, device=device, generator=generator)
        k = torch.randn(tokens, kv_heads * head_dim, device=device, generator=generator)
        v = torch.randn(tokens, kv_heads * head_dim, device=device, generator=generator)
        if q_spike > 0.0:
            # One dominant channel per head: a within-head transform that
            # spreads the spike lowers the 64-block amax and must improve the
            # true-path output MSE, so the real gate should accept.
            q = q * 0.02
            q[:, 0::head_dim] = (
                torch.randn(tokens, q_heads, device=device, generator=generator)
                * q_spike
            )
        windows.append({
            "q": (q, torch.ones(tokens, q_heads * head_dim // 16, device=device)),
            "k": (k, torch.ones(tokens, kv_heads * head_dim // 16, device=device)),
            "v": (v, torch.ones(tokens, kv_heads * head_dim // 16, device=device)),
        })
    return windows


def base_states(q_heads: int = 4, kv_heads: int = 2, head_dim: int = 64):
    common = {
        "multiplier": None,
        "permutation": None,
        "importance": None,
        "offsets": torch.tensor((-1, 1, 2, 3, 4), dtype=torch.int8),
        "error_threshold": 1e-7,
        "accept_margin": 0.0,
        "max_refine_ratio": 0.0,
        "max_refine_blocks": 0,
        "version": 2,
    }
    return {
        "q_state": dict(common, num_heads=q_heads, head_dim=head_dim),
        "k_state": dict(common, center_mode=0, num_heads=kv_heads, head_dim=head_dim),
        "v_state": dict(common, num_heads=kv_heads, head_dim=head_dim),
    }


def main():
    candidate = load(HERE / "candidate" / "solution.py", "a22_verify_candidate")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    result = {
        "candidate_sha256": hashlib.sha256(
            (HERE / "candidate" / "solution.py").read_bytes()
        ).hexdigest(),
        "parent_sha256": hashlib.sha256((ROOT / "solution.py").read_bytes()).hexdigest(),
    }

    # Real 32-step training on the identity arm: with parent rotations equal
    # to I, tq = exp(S) and tk = exp(-S) directly, so tq @ tk^T == I is the
    # reciprocal relation itself.
    states64 = base_states(head_dim=64)
    windows64 = make_windows(device, head_dim=64)
    tq, tk, center, train_info = candidate._a22b_train(
        windows64, states64, 4, 2, 64, device
    )
    assert tuple(tq.shape) == (2, 64, 64)
    assert tuple(tk.shape) == (2, 64, 64)
    assert center is None
    assert train_info["a22b_steps"] == 32
    assert train_info["a22b_fit_windows"] == 5
    assert train_info["a22b_inverse_error"] < 2e-4
    assert bool(torch.isfinite(tq).all() and torch.isfinite(tk).all())
    eye = torch.eye(64, device=device).expand(2, 64, 64)
    reciprocal_error = float(
        (tq.to(device) @ tk.to(device).transpose(-1, -2) - eye).abs().max()
    )
    assert reciprocal_error < 2e-4
    result["fixed_32_step_train"] = "PASS"
    result["reciprocal_exp_s_exp_minus_s_max_err"] = reciprocal_error

    # K-center synchronized compile: with rq = rk = I, tk = exp(-S), so the
    # compiled center must equal c @ tk exactly (same exp(-S)).
    identity = torch.eye(64).expand(2, 64, 64).clone()
    center64 = torch.randn(2, 64) * 0.05
    states_center = base_states(head_dim=64)
    states_center["q_state"]["learned_rotation"] = identity.clone()
    states_center["k_state"]["learned_rotation"] = identity.clone()
    states_center["k_state"]["learned_center"] = center64.clone()
    tq1, tk1, c1, info1 = candidate._a22b_train(
        windows64, states_center, 4, 2, 64, device
    )
    assert info1["a22b_s_norm"] > 0.0  # training actually moved S
    assert info1["a22b_center_compiled"] is True
    expected_center = (center64.unsqueeze(-2) @ tk1).squeeze(-2)
    torch.testing.assert_close(c1, expected_center, atol=1e-5, rtol=1e-5)
    result["k_center_sync_compile"] = "PASS"

    # S=0 must reproduce the parent exactly, center included.
    tq0, tk0, c0, info0 = candidate._a22b_train(
        windows64[:1], states_center, 4, 2, 64, device, force_zero=True
    )
    torch.testing.assert_close(tq0, identity, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(tk0, identity, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(c0, center64, atol=1e-6, rtol=1e-6)
    assert info0["a22b_s_norm"] == 0.0
    result["zero_residual_parent_control"] = "PASS"

    # Real end-to-end run of the overridden calibration: cheap mocked parent
    # calibration (identity arm) but the REAL 32-step trainer and the REAL
    # true-path gate.  Spike data makes the residual genuinely useful, so the
    # gate must accept; this proves the training branch executes and is
    # adopted, with attempted/accepted counters recorded.  5 windows also
    # pins the original split: fit = 4 windows, gate = last window only.
    spike_windows = make_windows(device, head_dim=64, q_spike=8.0)
    original_parent = candidate._A21_PARENT_CALIBRATION
    candidate._A21_PARENT_CALIBRATION = lambda *args, **kwargs: base_states(head_dim=64)
    try:
        with torch.inference_mode():
            accepted = candidate.hif4_calibration_attention(spike_windows, 4, 2, 64)
    finally:
        candidate._A21_PARENT_CALIBRATION = original_parent

    assert accepted["q_state"]["a22b_arm"] == "accepted", accepted["q_state"].get("a22b_gate")
    assert accepted["q_state"]["a22b_attempted"] == 1
    assert accepted["q_state"]["a22b_accepted"] == 1
    assert accepted["q_state"]["a22b_fit_windows"] == 4
    assert accepted["q_state"]["a22b_gate_windows"] == 1
    gate = accepted["q_state"]["a22b_gate"]
    assert len(gate) == 1 and gate[0]["window"] == 4 and gate[0]["pass"]
    assert gate[0]["candidate_mse"] < gate[0]["parent_mse"]
    result["real_training_branch_adopted"] = "PASS"
    result["gate_parent_mse"] = gate[0]["parent_mse"]
    result["gate_candidate_mse"] = gate[0]["candidate_mse"]

    # Parent fallback: real trainer, but the gate reports the candidate as
    # strictly worse, so the parent state must be returned untouched.
    original_gate = candidate._a21_gate_loss
    gate_calls = {"count": 0}

    def fake_gate(*args, **kwargs):
        gate_calls["count"] += 1
        return 1.0 if gate_calls["count"] % 2 == 1 else 1.1

    candidate._A21_PARENT_CALIBRATION = lambda *args, **kwargs: base_states(head_dim=64)
    candidate._a21_gate_loss = fake_gate
    try:
        with torch.inference_mode():
            rejected = candidate.hif4_calibration_attention(spike_windows, 4, 2, 64)
    finally:
        candidate._A21_PARENT_CALIBRATION = original_parent
        candidate._a21_gate_loss = original_gate

    assert rejected["q_state"]["a22b_arm"] == "parent"
    assert rejected["q_state"]["a22b_attempted"] == 1
    assert rejected["q_state"]["a22b_accepted"] == 0
    assert "learned_rotation" not in rejected["q_state"]
    result["parent_fallback"] = "PASS"

    # Legal state and finite outputs through the deployed dynamic APIs.
    for state, role, heads in (
        (accepted["q_state"], "q", 4),
        (accepted["k_state"], "k", 2),
        (accepted["v_state"], "v", 2),
    ):
        ref.validate_state(state)
        params = getattr(candidate, "hif4_dynamic_quantize_" + role)(
            *spike_windows[0][role], heads, 64, state
        )
        ref.validate_hif4_params(params, spike_windows[0][role][0].shape)
        assert bool(torch.isfinite(candidate._dequantize_hif4(params)).all())
    result["state_and_output_contract"] = "PASS"

    (HERE / "verification.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
