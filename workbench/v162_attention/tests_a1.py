"""A1 necessary tests for the v162-independent attention rotation candidate.

Run with the repo venv python while holding the v162-independent GPU lock.
Covers the workpackage A1 checklist:
  T1  R=I alignment with v162 (rotation path and identity arm, bitwise)
  T2  synthetic orthogonal R keeps continuous GQA QK products
  T3  token-count / call-order independence, no cross-call state mutation
  T4  five-field legality + decoded equals the research hard forward
  T5  STE surrogate gradient finite/nonzero; training smoke produces valid state
  T6  Linear and V paths bitwise identical to v162
"""

from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
BASELINE_PATH = ROOT / "workbench/v162_attention/baseline/solution.py"
CANDIDATE_PATH = ROOT / "workbench/v162_attention/candidate/solution.py"
REFERENCE_PATH = ROOT / "evaluator/reference_hif4.py"

FP4_GRID = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


baseline = load_module(BASELINE_PATH, "v162_attention_baseline")
candidate = load_module(CANDIDATE_PATH, "v162_attention_candidate")
sys.path.insert(0, str(REFERENCE_PATH.parent))
import reference_hif4 as ref  # noqa: E402


def make_pair(tokens: int, channels: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Synthetic NVFP4-shaped (quant, scale) pair with block scales."""

    generator = torch.Generator().manual_seed(seed)
    dense = torch.randn(tokens, channels, generator=generator)
    blocks = channels // 16
    grouped = dense.unflatten(-1, (blocks, 16))
    scale = (grouped.abs().amax(-1) / 6.0).clamp(min=1e-6)
    normalized = grouped / scale[..., None]
    sign = normalized.sign()
    index = torch.argmin((normalized.abs()[..., None] - FP4_GRID).abs(), dim=-1)
    quant = (sign * FP4_GRID[index]).flatten(-2, -1)
    return quant, scale


def synthetic_windows(windows: int, lengths: list[int], seed: int = 7):
    items = []
    for index, length in enumerate(lengths):
        items.append({
            "q": make_pair(length, 896, seed * 100 + index),
            "k": make_pair(length, 128, seed * 200 + index),
            "v": make_pair(length, 128, seed * 300 + index),
        })
    return items


def random_orthogonal(dim: int, groups: int, seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    matrices = []
    for _ in range(groups):
        a = torch.randn(dim, dim, generator=generator)
        q, r = torch.linalg.qr(a)
        matrices.append(q * torch.sign(torch.diag(r))[None, :])
    return torch.stack(matrices).to(torch.float32)


def assert_fields_equal(left: dict, right: dict, label: str) -> None:
    for key in ("scale_factor", "scale_lv2", "scale_lv3", "sign", "mant"):
        if not torch.equal(left[key], right[key]):
            raise AssertionError(f"{label}: field {key} differs from v162")


def test_t1_identity_alignment() -> None:
    eye = torch.eye(64)[None].repeat(2, 1, 1)
    state = {
        "format": "group-rotation-v1", "arm": "manual", "mode": "rotation",
        "q_heads": 14, "kv_heads": 2, "head_dim": 64,
        "gate_loss_identity": 1.0, "gate_loss_h": -1.0,
        "gate_loss_learned": -1.0, "trained_steps": 0,
        "r": eye.clone(),
    }
    q_quant, q_scale = make_pair(33, 896, 11)
    k_quant, k_scale = make_pair(33, 128, 12)
    v_quant, v_scale = make_pair(33, 128, 13)
    base_q = baseline.hif4_dynamic_quantize_q(q_quant, q_scale, 14, 64, {})
    cand_q = candidate.hif4_dynamic_quantize_q(q_quant, q_scale, 14, 64, state)
    assert_fields_equal(cand_q, base_q, "T1 dynamic Q with R=I")
    base_k = baseline.hif4_dynamic_quantize_k(k_quant, k_scale, 2, 64, {})
    cand_k = candidate.hif4_dynamic_quantize_k(k_quant, k_scale, 2, 64, state)
    assert_fields_equal(cand_k, base_k, "T1 dynamic K with R=I")

    saved_arm = candidate._ROT_ARM
    candidate._ROT_ARM = "identity"
    try:
        states = candidate.hif4_calibration_attention(
            synthetic_windows(3, [10, 24, 40]), 14, 2, 64
        )
    finally:
        candidate._ROT_ARM = saved_arm
    ref.validate_state(states["q_state"])
    ref.validate_state(states["k_state"])
    ref.validate_state(states["v_state"])
    cand_q2 = candidate.hif4_dynamic_quantize_q(
        q_quant, q_scale, 14, 64, states["q_state"]
    )
    assert_fields_equal(cand_q2, base_q, "T1 identity-arm dynamic Q")
    print("T1 identity alignment: PASS")


def test_t2_continuous_gqa_invariance() -> None:
    rotation = random_orthogonal(64, 2, 23)
    generator = torch.Generator().manual_seed(24)
    q = torch.randn(48, 896, generator=generator)
    k = torch.randn(64, 128, generator=generator)
    q_rot = candidate._rotate_rows(q, 14, rotation)
    k_rot = candidate._rotate_rows(k, 2, rotation)
    group_of = torch.arange(14) // 7
    logits = torch.stack(
        [q.reshape(48, 14, 64)[:, h, :] @ k.reshape(64, 2, 64)[:, group_of[h], :].T
         for h in range(14)], dim=-1
    )
    logits_rot = torch.stack(
        [q_rot.reshape(48, 14, 64)[:, h, :] @ k_rot.reshape(64, 2, 64)[:, group_of[h], :].T
         for h in range(14)], dim=-1
    )
    error = float((logits_rot - logits).abs().max())
    bound = 1e-5 + 1e-5 * float(logits.abs().max())
    if error > bound:
        raise AssertionError(f"T2 continuous QK error {error} exceeds bound {bound}")
    print(f"T2 continuous GQA invariance: PASS (max err {error:.2e} <= {bound:.2e})")


def test_t3_order_and_token_counts() -> None:
    rotation = random_orthogonal(64, 2, 31)
    state = {
        "format": "group-rotation-v1", "arm": "manual", "mode": "rotation",
        "q_heads": 14, "kv_heads": 2, "head_dim": 64,
        "gate_loss_identity": 1.0, "gate_loss_h": -1.0,
        "gate_loss_learned": -1.0, "trained_steps": 0,
        "r": rotation.clone(),
    }
    q_long, q_long_scale = make_pair(91, 896, 41)
    k_short, k_short_scale = make_pair(37, 128, 42)
    first_q = candidate.hif4_dynamic_quantize_q(q_long, q_long_scale, 14, 64, state)
    first_k = candidate.hif4_dynamic_quantize_k(k_short, k_short_scale, 2, 64, state)
    second_k = candidate.hif4_dynamic_quantize_k(k_short, k_short_scale, 2, 64, state)
    second_q = candidate.hif4_dynamic_quantize_q(q_long, q_long_scale, 14, 64, state)
    assert_fields_equal(first_q, second_q, "T3 Q order independence")
    assert_fields_equal(first_k, second_k, "T3 K order independence")
    q_again, _ = make_pair(91, 896, 41)
    assert torch.equal(q_again, q_long), "T3 input mutated across calls"
    q_tiny, q_tiny_scale = make_pair(3, 896, 43)
    out_tiny = candidate.hif4_dynamic_quantize_q(q_tiny, q_tiny_scale, 14, 64, state)
    ref.validate_hif4_params(out_tiny, q_tiny.shape)
    print("T3 token counts and call order: PASS")


def test_t4_legality_and_surrogate_forward() -> None:
    rotation = random_orthogonal(64, 2, 51)
    state = {
        "format": "group-rotation-v1", "arm": "manual", "mode": "rotation",
        "q_heads": 14, "kv_heads": 2, "head_dim": 64,
        "gate_loss_identity": 1.0, "gate_loss_h": -1.0,
        "gate_loss_learned": -1.0, "trained_steps": 0,
        "r": rotation.clone(),
    }
    q_quant, q_scale = make_pair(70, 896, 61)
    k_quant, k_scale = make_pair(55, 128, 62)
    for api, quant, scale, heads in (
        ("q", q_quant, q_scale, 14), ("k", k_quant, k_scale, 2),
    ):
        params = (
            candidate.hif4_dynamic_quantize_q(quant, scale, heads, 64, state)
            if api == "q"
            else candidate.hif4_dynamic_quantize_k(quant, scale, heads, 64, state)
        )
        ref.validate_hif4_params(params, quant.shape)
        decoded = ref.dequantize_hif4(
            {k: v.clone() for k, v in params.items()}, quant.shape
        )
        dense = candidate.dequantize_nvfp4(quant, scale).to(torch.float32)
        rows = dense.reshape(-1, dense.shape[-1])
        rotated = candidate._rotate_rows(rows, heads, rotation).reshape(dense.shape)
        manual = candidate._decode_hif5_fields(candidate._encode_standard_hif4(rotated))
        if not torch.equal(decoded, manual):
            diff = float((decoded - manual).abs().max())
            raise AssertionError(f"T4 {api} decoded != research forward (max diff {diff})")
    print("T4 legality + decoded equals research forward: PASS")


def test_t5_surrogate_gradient_and_training() -> None:
    x = torch.randn(16, 896) * 0.3
    x.requires_grad_(True)
    ste = candidate._ste_encode(x)
    hard = candidate._decode_hif5_fields(candidate._encode_standard_hif4(x.detach()))
    if not torch.equal(ste.detach(), hard):
        raise AssertionError("T5 STE forward != decode(encode(x))")
    ste.sum().backward()
    if x.grad is None or not bool(torch.isfinite(x.grad).all()) or float(x.grad.abs().sum()) == 0.0:
        raise AssertionError("T5 STE gradient missing, non-finite, or zero")
    windows = [
        {name: value for name, value in window.items()}
        for window in _decoded_synthetic_windows([24, 32])
    ]
    rotation, info = candidate._train_rotation(windows, 14, 2, 64, torch.device("cpu"))
    if not bool(torch.isfinite(rotation).all()):
        raise AssertionError("T5 trained rotation has non-finite values")
    if info["ortho_error"] > 1e-3:
        raise AssertionError(f"T5 orthogonality error too large: {info['ortho_error']}")
    saved_arm = candidate._ROT_ARM
    candidate._ROT_ARM = "learned"
    try:
        states = candidate.hif4_calibration_attention(
            _raw_synthetic_windows([24, 32]), 14, 2, 64
        )
    finally:
        candidate._ROT_ARM = saved_arm
    ref.validate_state(states["q_state"])
    ref.validate_state(states["k_state"])
    ref.validate_state(states["v_state"])
    if states["q_state"].get("mode") != "rotation":
        raise AssertionError("T5 learned arm must deploy the rotation unconditionally")
    print(
        f"T5 surrogate gradient + training smoke: PASS "
        f"(train loss {info['final_train_loss']:.4f}, ortho {info['ortho_error']:.2e})"
    )


def test_t6_linear_v_bitwise() -> None:
    for rows, channels, seed in ((896, 896, 71), (128, 896, 72), (4864, 896, 73)):
        w_quant, w_scale = make_pair(rows, channels, seed)
        a_quant, a_scale = make_pair(20, channels, seed + 50)
        base = baseline.hif4_calibration_and_quantize_weight(w_quant, w_scale, [])
        cand = candidate.hif4_calibration_and_quantize_weight(w_quant, w_scale, [])
        assert_fields_equal(cand["weight_params"], base["weight_params"], f"T6 weight {rows}")
        assert cand["activation_state"] == base["activation_state"] == {}
        base_act = baseline.hif4_dynamic_quantize_activation(a_quant, a_scale, {})
        cand_act = candidate.hif4_dynamic_quantize_activation(a_quant, a_scale, {})
        assert_fields_equal(cand_act, base_act, f"T6 activation rows={rows}")
    v_quant, v_scale = make_pair(29, 128, 81)
    base_v = baseline.hif4_dynamic_quantize_v(v_quant, v_scale, 2, 64, {})
    cand_v = candidate.hif4_dynamic_quantize_v(v_quant, v_scale, 2, 64, {})
    assert_fields_equal(cand_v, base_v, "T6 dynamic V")
    print("T6 Linear/V bitwise identity: PASS")


def _decoded_synthetic_windows(lengths: list[int]):
    raw = _raw_synthetic_windows(lengths)
    return [candidate._decode_window(item) for item in raw]


def _raw_synthetic_windows(lengths: list[int]):
    return synthetic_windows(len(lengths), lengths, seed=3)


if __name__ == "__main__":
    torch.manual_seed(0)
    test_t1_identity_alignment()
    test_t2_continuous_gqa_invariance()
    test_t3_order_and_token_counts()
    test_t4_legality_and_surrogate_forward()
    test_t5_surrogate_gradient_and_training()
    test_t6_linear_v_bitwise()
    print("ALL A1 TESTS PASSED")
