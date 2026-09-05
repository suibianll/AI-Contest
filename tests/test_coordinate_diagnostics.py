"""Algebraic and mirror-consistency tests for coordinate-consistent diagnostics.

The synthetic tests run in FP64/FP32 without the real model: they verify the
two decomposition identities, the fixed-coordinate Attention arm semantics,
and that the Linear continuous mirrors (weight/activation) reproduce the
unquantized product under smooth/permute/block-hadamard and a u-perp-v rank
residual.  Real-chain (case-level) reproduction is not part of this unit test;
it is checked by the CLI run against the recorded player gain of the same SHA.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "evaluator"))

import solution  # noqa: E402
import coordinate_diagnostics as diag  # noqa: E402


def _rel(a: torch.Tensor, b: torch.Tensor) -> float:
    denom = float((b * b).sum())
    if denom <= 0.0:
        return float(((a - b) * (a - b)).sum())
    return float(((a - b) * (a - b)).sum() / denom) ** 0.5


# --------------------------------------------------------------------------- #
# Identity tests (FP64)
# --------------------------------------------------------------------------- #
def test_linear_arms_expansion_identity_fp64() -> None:
    torch.manual_seed(11)
    dim, out = 40, 7
    x_t = torch.randn(6, dim, dtype=torch.float64)
    w_t = torch.randn(out, dim, dtype=torch.float64)
    e_x = torch.randn(6, dim, dtype=torch.float64) * 0.2
    e_w = torch.randn(out, dim, dtype=torch.float64) * 0.2
    ref = x_t @ w_t.t()
    info = diag.linear_arms(x_t, w_t, x_t + e_x, w_t + e_w, ref)
    exp = info["expansion"]
    lhs = exp["lhs_ms_Yhh_minus_Ytt"]
    rhs = exp["rhs_exact"]
    scale = max(abs(lhs), abs(rhs), 1e-30)
    assert abs(lhs - rhs) / scale <= 1e-10
    # X_hW_h must coincide with the directly computed player product.
    player_mse = info["mse_to_ref"]["X_hW_h"]
    direct = float(((x_t + e_x) @ (w_t + e_w).t() - ref).square().mean())
    assert abs(player_mse - direct) <= 1e-12 * max(1.0, abs(direct))


def test_attention_be_decomposition_identity_fp64() -> None:
    torch.manual_seed(12)
    seq, dim, q_heads, kv_heads = 8, 8, 1, 1
    q = torch.randn(seq, dim, dtype=torch.float64)
    k = torch.randn(seq, dim, dtype=torch.float64)
    v = torch.randn(seq, dim, dtype=torch.float64)
    ref = v2_attention_ref(q, k, v, q_heads, kv_heads, dim)
    q_h = q + torch.randn_like(q) * 0.1
    k_h = k + torch.randn_like(k) * 0.1
    v_h = v + torch.randn_like(v) * 0.1
    info = diag.attention_arms(
        q, k, v, q_h, k_h, v_h, q_heads, kv_heads, dim, ref
    )
    be = info["be_decomposition"]
    scale = max(abs(be["lhs_mse_oh_ref"]), abs(be["rhs_sum"]), 1e-30)
    assert abs(be["lhs_mse_oh_ref"] - be["rhs_sum"]) / scale <= 1e-10
    # Player MSE equals the directly computed quantity.
    assert abs(info["mse_player_vs_ref"] - info["mse_to_ref"]["111"]) <= 1e-30


def test_attention_arms_degenerate_and_symmetric() -> None:
    """With GQA (q_heads=2, kv_heads=1) and causal masking."""
    torch.manual_seed(13)
    seq, dim, q_heads, kv_heads = 16, 16, 2, 1
    q = torch.randn(seq, q_heads * dim)
    k = torch.randn(seq, kv_heads * dim)
    v = torch.randn(seq, kv_heads * dim)
    ref = v2_attention_ref(q, k, v, q_heads, kv_heads, dim)
    # Degenerate: identical float/quantised operands -> every arm identical.
    info = diag.attention_arms(
        q, k, v, q.clone(), k.clone(), v.clone(), q_heads, kv_heads, dim, ref
    )
    assert info["mse_player_vs_t"] == 0.0
    for arm in ("100", "010", "001", "110", "101", "011", "111"):
        assert info["mse_to_ref"][arm] == pytest.approx(info["mse_to_ref"]["000"])
    # K-identical V-only substitution: 001 == 011 by construction.
    v_h = v + torch.randn_like(v) * 0.05
    info = diag.attention_arms(
        q, k, v, q, k.clone(), v_h, q_heads, kv_heads, dim, ref
    )
    assert info["mse_to_t"]["001"] > 0.0
    assert info["mse_to_t"]["011"] == pytest.approx(info["mse_to_t"]["001"])
    # V-only substitution leaves Q-only / K-only arms untouched.
    assert info["mse_to_t"]["100"] == 0.0
    assert info["mse_to_t"]["010"] == 0.0


def v2_attention_ref(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q_heads: int,
    kv_heads: int,
    head_dim: int,
) -> torch.Tensor:
    import official_eval as v2
    return v2._attention(q[None], k[None], v[None], q_heads, kv_heads, head_dim)


# --------------------------------------------------------------------------- #
# Linear continuous mirrors (FP32, mirroring solution.py)
# --------------------------------------------------------------------------- #
def test_linear_continuous_preserves_product() -> None:
    torch.manual_seed(17)
    in_features, out_features = 64, 12
    x = torch.randn(9, in_features) * 0.2
    w = torch.randn(out_features, in_features) * 0.2

    d = torch.exp(torch.randn(in_features) * 0.15)
    smooth_inv = 1.0 / d
    perm = torch.randperm(in_features)
    bss, seed = 4, 1
    ref = x @ w.t()

    # (1) Smooth + permutation + block hadamard, no residual.
    state: dict = {
        "smooth_inv": smooth_inv,
        "permutation": perm,
        "block_smooth_size": bss,
        "block_smooth_seed": seed,
    }
    x_t = diag.linear_activation_continuous(x, state, solution)
    w_t = diag.linear_weight_continuous(w, state, solution)
    assert _rel(x_t @ w_t.t(), ref) <= 2e-4

    # (2) Add a non-trivial rank-1 residual with u perpendicular to v.
    #     activation_state stores rank1_u/rank1_v as 1-D vectors ([in_features]).
    v1 = torch.randn(in_features)
    v1 = v1 / v1.norm()
    u1 = torch.randn(in_features)
    u1 = u1 - float(u1 @ v1) * v1  # orthogonalise
    u1 = u1 / u1.norm()
    state["rank1_u"] = u1
    state["rank1_v"] = v1
    x_t = diag.linear_activation_continuous(x, state, solution)
    w_t = diag.linear_weight_continuous(w, state, solution)
    assert _rel(x_t @ w_t.t(), ref) <= 5e-4

    # (3) rank-2 fused path (residual_u/residual_v as [in_features, 2]).
    state["residual_u"] = torch.stack([u1, torch.zeros_like(u1)], dim=1)
    state["residual_v"] = torch.stack([v1, torch.zeros_like(v1)], dim=1)
    del state["rank1_u"], state["rank1_v"]
    x_t = diag.linear_activation_continuous(x, state, solution)
    w_t = diag.linear_weight_continuous(w, state, solution)
    assert torch.isfinite(x_t).all() and torch.isfinite(w_t).all()
    assert _rel(x_t @ w_t.t(), ref) <= 5e-4


def test_linear_identity_state_mirror_is_raw() -> None:
    torch.manual_seed(18)
    x = torch.randn(5, 64) * 0.3
    w = torch.randn(6, 64) * 0.3
    # No smooth / permutation / block / residual fields: mirrors must be raw.
    state: dict = {}
    x_t = diag.linear_activation_continuous(x, state, solution)
    w_t = diag.linear_weight_continuous(w, state, solution)
    assert torch.equal(x_t, x)
    assert torch.equal(w_t, w)
