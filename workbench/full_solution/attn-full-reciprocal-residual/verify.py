"""Independent reciprocal-compile, training, reachability, and fallback checks."""

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
                 head_dim: int = 64, tokens: int = 8):
    windows = []
    for index in range(5):
        generator = torch.Generator(device=device).manual_seed(21071 + index)
        q = torch.randn(tokens, q_heads * head_dim, device=device, generator=generator)
        k = torch.randn(tokens, kv_heads * head_dim, device=device, generator=generator)
        v = torch.randn(tokens, kv_heads * head_dim, device=device, generator=generator)
        windows.append({
            "q": (q, torch.ones(tokens, q_heads * head_dim // 16, device=device)),
            "k": (k, torch.ones(tokens, kv_heads * head_dim // 16, device=device)),
            "v": (v, torch.ones(tokens, kv_heads * head_dim // 16, device=device)),
        })
    return windows


def base_states(q_heads: int = 4, kv_heads: int = 2, head_dim: int = 256):
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
    candidate = load(HERE / "candidate" / "solution.py", "a21_verify_candidate")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    result = {
        "candidate_sha256": hashlib.sha256(
            (HERE / "candidate" / "solution.py").read_bytes()
        ).hexdigest(),
        "parent_sha256": hashlib.sha256((ROOT / "solution.py").read_bytes()).hexdigest(),
    }

    # A small real training run checks the fixed 32-step residual trainer and
    # its reciprocal compile, without using historical parent code.
    states64 = base_states(head_dim=64)
    windows64 = make_windows(device, head_dim=64)
    tq, tk, center, train_info = candidate._a22b_train(
        windows64, states64, 4, 2, 64, device
    )
    assert tuple(tq.shape) == (2, 64, 64)
    assert tuple(tk.shape) == (2, 64, 64)
    assert center is None
    assert train_info["a22b_steps"] == 32
    assert train_info["a22b_inverse_error"] < 2e-4
    assert bool(torch.isfinite(tq).all() and torch.isfinite(tk).all())
    result["fixed_32_step_train_and_reciprocal_compile"] = "PASS"

    # S=0 must be an exact parent control, including an additive learned K
    # center compiled through exp(-S).
    identity = torch.eye(64).expand(2, 64, 64).clone()
    center64 = torch.randn(2, 64) * 0.02
    states_center = base_states(head_dim=64)
    states_center["q_state"]["learned_rotation"] = identity.clone()
    states_center["k_state"]["learned_rotation"] = identity.clone()
    states_center["k_state"]["learned_center"] = center64.clone()
    tq0, tk0, c0, info0 = candidate._a22b_train(
        windows64[:1], states_center, 4, 2, 64, device, force_zero=True
    )
    torch.testing.assert_close(tq0, identity, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(tk0, identity, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(c0, center64, atol=1e-6, rtol=1e-6)
    assert info0["a22b_s_norm"] == 0.0
    result["zero_residual_parent_control"] = "PASS"

    # Exercise both the accepted and parent-fallback branches independently
    # of the real calibration parent, while retaining the real state contract.
    windows256 = make_windows(device, head_dim=256)
    original_parent = candidate._A21_PARENT_CALIBRATION
    original_train = candidate._a22b_train
    original_gate = candidate._a21_gate_loss
    candidate._A21_PARENT_CALIBRATION = lambda *args, **kwargs: base_states(head_dim=256)
    identity256 = torch.eye(256).expand(2, 256, 256).clone()
    candidate._a22b_train = lambda *args, **kwargs: (
        identity256.clone(), identity256.clone(), None,
        {"a22b_steps": 32, "a22b_inverse_error": 0.0, "a22b_s_norm": 0.0},
    )
    gate_calls = {"count": 0}
    accept_mode = {"value": True}

    def fake_gate(*args, **kwargs):
        gate_calls["count"] += 1
        if gate_calls["count"] % 2 == 1:
            return 1.0
        return 0.9 if accept_mode["value"] else 1.1

    candidate._a21_gate_loss = fake_gate
    try:
        with torch.inference_mode():
            accepted = candidate.hif4_calibration_attention(windows256, 4, 2, 256)
            accept_mode["value"] = False
            gate_calls["count"] = 0
            rejected = candidate.hif4_calibration_attention(windows256, 4, 2, 256)
    finally:
        candidate._A21_PARENT_CALIBRATION = original_parent
        candidate._a22b_train = original_train
        candidate._a21_gate_loss = original_gate

    assert accepted["q_state"]["a22b_arm"] == "accepted"
    assert accepted["q_state"]["a22b_attempted"] == 1
    assert accepted["q_state"]["a22b_accepted"] == 1
    assert accepted["q_state"]["a22b_gate_windows"] == 2
    assert torch.equal(accepted["q_state"]["learned_rotation"], identity256)
    result["forced_reachability"] = "PASS"

    assert rejected["q_state"]["a22b_arm"] == "parent"
    assert rejected["q_state"]["a22b_attempted"] == 1
    assert rejected["q_state"]["a22b_accepted"] == 0
    assert "learned_rotation" not in rejected["q_state"]
    result["parent_fallback"] = "PASS"

    for state, role, heads in (
        (accepted["q_state"], "q", 4),
        (accepted["k_state"], "k", 2),
        (accepted["v_state"], "v", 2),
    ):
        ref.validate_state(state)
        params = getattr(candidate, "hif4_dynamic_quantize_" + role)(
            *windows256[0][role], heads, 256, state
        )
        ref.validate_hif4_params(params, windows256[0][role][0].shape)
        assert bool(torch.isfinite(candidate._dequantize_hif4(params)).all())
    result["state_and_output_contract"] = "PASS"

    (HERE / "verification.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
