"""Independent reachability, parent-fallback, and deployment checks."""

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
                 head_dim: int = 256, tokens: int = 8):
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
    candidate = load(HERE / "candidate" / "solution.py", "tri_verify_candidate")
    parent = load(ROOT / "solution.py", "tri_verify_parent")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    result = {
        "candidate_sha256": hashlib.sha256(
            (HERE / "candidate" / "solution.py").read_bytes()
        ).hexdigest(),
        "parent_sha256": hashlib.sha256((ROOT / "solution.py").read_bytes()).hexdigest(),
    }

    # Check the analytic Attention backward against autograd for both masks.
    torch.manual_seed(21071)
    q = torch.randn(4, 4 * 256, requires_grad=True)
    k = torch.randn(4, 2 * 256, requires_grad=True)
    v = torch.randn(4, 2 * 256)
    target = torch.randn(4, 4 * 256)
    for causal in (True, False):
        output = candidate._attention_forward(q, k, v, 4, 2, 256, causal)
        loss = (output - target).square().mean()
        exact_q, exact_k = torch.autograd.grad(loss, (q, k), retain_graph=True)
        d_output = 2.0 * (output.detach() - target) / float(output.numel())
        found_q, found_k = candidate._tri_attention_backward(
            d_output, q.detach(), k.detach(), v, 4, 2, 256, causal
        )
        torch.testing.assert_close(found_q, exact_q, atol=3e-5, rtol=3e-5)
        torch.testing.assert_close(found_k, exact_k, atol=3e-5, rtol=3e-5)
    result["analytic_attention_backward"] = "PASS"

    # Compile a deterministic rank-1 proposal and check exact QK invariance.
    blocks = torch.zeros(2, 2, 64, 64)
    blocks[0, 0] = torch.outer(torch.ones(64), torch.arange(64, dtype=torch.float32)) * 1e-4
    blocks[1, 1] = torch.outer(torch.arange(64, dtype=torch.float32), torch.ones(64)) * 1e-4
    q_plain = torch.randn(5, 4 * 256)
    k_plain = torch.randn(5, 2 * 256)
    q_transport = candidate._apply_attn_triangular_transform(q_plain, 4, blocks, False)
    k_transport = candidate._apply_attn_triangular_transform(k_plain, 2, blocks, True)
    qh_plain = q_plain.reshape(5, 4, 256)
    kh_plain = k_plain.reshape(5, 2, 256).repeat_interleave(2, dim=1)
    qh_transport = q_transport.reshape(5, 4, 256)
    kh_transport = k_transport.reshape(5, 2, 256).repeat_interleave(2, dim=1)
    torch.testing.assert_close(
        torch.einsum("thi,shi->ths", qh_plain, kh_plain),
        torch.einsum("thi,shi->ths", qh_transport, kh_transport),
        atol=3e-5,
        rtol=3e-5,
    )
    result["rank1_triangular_qk_invariance"] = "PASS"

    windows = make_windows(device)
    original_parent = candidate._TRI_PARENT_CALIBRATION
    original_gradient = candidate._attn_block_output_gradient
    original_boundary = candidate._attn_first_code_boundary_step
    original_loss = candidate._tri_true_output_loss
    fake_parent_states = base_states()
    fake_gradient = torch.zeros(2, 2, 64, 64)
    fake_gradient[:, :, torch.arange(64), torch.arange(64)] = 1.0
    force_pass = True

    candidate._TRI_PARENT_CALIBRATION = lambda *args, **kwargs: base_states()
    candidate._attn_block_output_gradient = lambda *args, **kwargs: (
        fake_gradient.clone(),
        {"tri_fit_windows": 3, "tri_fit_masks": 6,
         "tri_singular_values": [[1.0, 1.0], [1.0, 1.0]]},
    )
    candidate._attn_first_code_boundary_step = lambda *args, **kwargs: (0.01, 1, 3, 2)

    def fake_loss(sample, states, *args, **kwargs):
        active = "triangular_blocks" in states["q_state"]
        value = 0.9 if active and force_pass else (1.1 if active else 1.0)
        return {"mean": value, "causal": value, "noncausal": value}

    candidate._tri_true_output_loss = fake_loss
    try:
        with torch.inference_mode():
            accepted = candidate.hif4_calibration_attention(windows, 4, 2, 256)
            force_pass = False
            rejected = candidate.hif4_calibration_attention(windows, 4, 2, 256)
    finally:
        candidate._TRI_PARENT_CALIBRATION = original_parent
        candidate._attn_block_output_gradient = original_gradient
        candidate._attn_first_code_boundary_step = original_boundary
        candidate._tri_true_output_loss = original_loss

    assert accepted["q_state"]["tri_arm"] == "accepted"
    assert accepted["q_state"]["tri_attempted"] == 1
    assert accepted["q_state"]["tri_accepted"] == 1
    assert accepted["q_state"]["tri_reachable_blocks"] == 4
    assert accepted["q_state"]["triangular_inverse"] is False
    assert accepted["k_state"]["triangular_inverse"] is True
    assert tuple(accepted["q_state"]["triangular_blocks"].shape) == (2, 2, 64, 64)
    assert bool(torch.isfinite(accepted["q_state"]["triangular_blocks"]).all())
    result["forced_reachability"] = "PASS"

    assert rejected["q_state"]["tri_arm"] == "parent"
    assert rejected["q_state"]["tri_attempted"] == 1
    assert rejected["q_state"]["tri_accepted"] == 0
    assert "triangular_blocks" not in rejected["q_state"]
    result["parent_fallback"] = "PASS"

    for state, role, heads in (
        (accepted["q_state"], "q", 4),
        (accepted["k_state"], "k", 2),
        (accepted["v_state"], "v", 2),
    ):
        ref.validate_state(state)
        params = getattr(candidate, "hif4_dynamic_quantize_" + role)(
            *windows[0][role], heads, 256, state
        )
        ref.validate_hif4_params(params, windows[0][role][0].shape)
        assert bool(torch.isfinite(candidate._dequantize_hif4(params)).all())
    result["state_and_output_contract"] = "PASS"

    (HERE / "verification.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
