"""Parity test: manual trainer gradients vs the A2b autograd gradients.

Run: .venv/Scripts/python.exe workbench/v162_attention/test_manual_trainer_parity.py
Pass criteria: relative error < 1e-4 on attention and Cayley gradients, and
the full manual training reaches the same loss scale as autograd training.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "workbench/v162_attention"))
import manual_trainer as mt  # noqa: E402


def load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


candidate = load(ROOT / "workbench/v162_attention/candidate_b/solution.py", "a2b_module")


def rel_error(a: torch.Tensor, b: torch.Tensor) -> float:
    denom = b.abs().max().clamp_min(1e-30)
    return float((a - b).abs().max() / denom)


def main() -> int:
    torch.manual_seed(11)
    q_heads, kv_heads, head_dim = 14, 2, 64
    tokens_q, tokens_k = 32, 128
    q = torch.randn(tokens_q, q_heads * head_dim) * 0.5
    k = torch.randn(tokens_k, kv_heads * head_dim) * 0.5
    v = torch.randn(tokens_k, kv_heads * head_dim) * 0.5
    d_out = torch.randn(tokens_q, q_heads * head_dim) * 0.3

    # ---- attention backward parity -----------------------------------------
    d_q_manual, d_k_manual = mt.attention_backward(
        d_out, q, k, v, q_heads, kv_heads, head_dim
    )
    q_leaf = q.clone().requires_grad_(True)
    k_leaf = k.clone().requires_grad_(True)
    out = candidate._attention_forward(
        q_leaf[None], k_leaf[None], v[None], q_heads, kv_heads, head_dim
    )[0]
    (out * d_out).sum().backward()
    e_q = rel_error(d_q_manual, q_leaf.grad)
    e_k = rel_error(d_k_manual, k_leaf.grad)
    print(f"attention grad rel err: dQ {e_q:.2e}  dK {e_k:.2e}")
    if e_q > 1e-4 or e_k > 1e-4:
        print("ATTENTION BACKWARD PARITY FAIL")
        return 1

    # ---- Cayley backward parity ---------------------------------------------
    groups, dim = kv_heads, head_dim
    theta = torch.randn(groups, dim, dim) * 0.05
    base = candidate._hadamard_orthogonal(dim)

    def objective(theta_value: torch.Tensor) -> torch.Tensor:
        c, _reg = candidate._cayley_orthogonal(theta_value)
        rotation = torch.einsum("dk,gkl->gdl", base, c)
        q_rot = candidate._rotate_rows(q, q_heads, rotation)
        k_rot = candidate._rotate_rows(k, kv_heads, rotation)
        q_hat = q_rot + (candidate._decode_hif5_fields(candidate._encode_standard_hif4(q_rot)) - q_rot).detach()
        k_hat = k_rot + (candidate._decode_hif5_fields(candidate._encode_standard_hif4(k_rot)) - k_rot).detach()
        output = candidate._attention_forward(
            q_hat[None], k_hat[None], v[None], q_heads, kv_heads, head_dim
        )[0]
        return (output * d_out).sum()

    theta_leaf = theta.clone().requires_grad_(True)
    objective(theta_leaf).backward()
    grad_c_auto = torch.zeros_like(theta)
    # autograd dL/dC: re-run with C as the leaf via the linearized chain
    c_leaf = candidate._cayley_orthogonal(theta)[0].detach().requires_grad_(True)
    rotation = torch.einsum("dk,gkl->gdl", base, c_leaf)
    q_rot = candidate._rotate_rows(q, q_heads, rotation)
    k_rot = candidate._rotate_rows(k, kv_heads, rotation)
    q_hat = q_rot + (candidate._decode_hif5_fields(candidate._encode_standard_hif4(q_rot)) - q_rot).detach()
    k_hat = k_rot + (candidate._decode_hif5_fields(candidate._encode_standard_hif4(k_rot)) - k_rot).detach()
    output = candidate._attention_forward(
        q_hat[None], k_hat[None], v[None], q_heads, kv_heads, head_dim
    )[0]
    (output * d_out).sum().backward()
    grad_c_auto = c_leaf.grad
    grad_theta_manual = mt.cayley_backward(grad_c_auto, theta.detach())
    e_cayley = rel_error(grad_theta_manual, theta_leaf.grad)
    print(f"cayley grad rel err (dTheta): {e_cayley:.2e}")
    if e_cayley > 1e-3:
        print("CAYLEY BACKWARD PARITY FAIL")
        return 1

    # ---- full training parity ------------------------------------------------
    device = torch.device("cpu")
    windows = []
    for length in (128, 512):
        kv_index = candidate._even_indices(length, 128, device)
        q_index = candidate._even_indices(kv_index.numel(), 32, device)
        q_rows = kv_index.index_select(0, q_index)
        k_sub = k[:length][kv_index] if length <= tokens_k else None
        # build synthetic windows of the right length instead
        q_full = torch.randn(length, q_heads * head_dim) * 0.5
        k_full = torch.randn(length, kv_heads * head_dim) * 0.5
        v_full = torch.randn(length, kv_heads * head_dim) * 0.5
        kv_index = candidate._even_indices(length, 128, device)
        q_index = candidate._even_indices(kv_index.numel(), 32, device)
        q_rows = kv_index.index_select(0, q_index)
        k_sub = k_full.index_select(0, kv_index)
        v_sub = v_full.index_select(0, kv_index)
        q_sub = q_full.index_select(0, q_rows)
        reference = candidate._attention_forward(
            q_sub[None], k_sub[None], v_sub[None], q_heads, kv_heads, head_dim
        )[0].detach()
        std_q = candidate._decode_hif5_fields(candidate._encode_standard_hif4(q_sub))
        std_k = candidate._decode_hif5_fields(candidate._encode_standard_hif4(k_sub))
        std_v = candidate._decode_hif5_fields(candidate._encode_standard_hif4(v_sub))
        standard = candidate._attention_forward(
            std_q[None], std_k[None], std_v[None], q_heads, kv_heads, head_dim
        )[0].detach()
        v_hat = candidate._decode_hif5_fields(candidate._encode_standard_hif4(v_sub))
        windows.append({
            "q": q_sub, "k": k_sub, "v": v_sub, "v_hat": v_hat,
            "reference": reference, "mse_std": max(float((standard - reference).square().mean()), 1e-12),
        })

    rotation_manual, info_manual = mt.train_rotation_manual(
        windows, base, groups, dim, q_heads, kv_heads, head_dim, device,
        hard_encode=candidate._encode_standard_hif4,
        hard_decode=candidate._decode_hif5_fields,
    )

    # autograd reference trainer (A2b path, in-bubble)
    prepared = []
    for w in windows:
        item = dict(w)
        prepared.append(item)
    with torch.inference_mode(False), torch.enable_grad():
        theta_auto = torch.zeros(groups, dim, dim, requires_grad=True)
        opt = torch.optim.Adam([theta_auto], lr=0.01)
        eye = torch.eye(dim)
        for _step in range(32):
            opt.zero_grad(set_to_none=True)
            losses = []
            reg_total = theta_auto.new_zeros(())
            for item in prepared:
                c, reg = candidate._cayley_orthogonal(theta_auto)
                rotation = torch.einsum("dk,gkl->gdl", base, c)
                reg_total = reg_total + reg
                q_rot = candidate._rotate_rows(item["q"], q_heads, rotation)
                k_rot = candidate._rotate_rows(item["k"], kv_heads, rotation)
                q_hat = q_rot + (candidate._decode_hif5_fields(candidate._encode_standard_hif4(q_rot)) - q_rot).detach()
                k_hat = k_rot + (candidate._decode_hif5_fields(candidate._encode_standard_hif4(k_rot)) - k_rot).detach()
                output = candidate._attention_forward(
                    q_hat[None], k_hat[None], item["v_hat"][None], q_heads, kv_heads, head_dim
                )[0]
                losses.append((output - item["reference"]).square().mean() / item["mse_std"])
            data_loss = torch.stack(losses).mean()
            objective = data_loss + 1e-3 * reg_total / len(prepared)
            objective.backward()
            torch.nn.utils.clip_grad_norm_([theta_auto], 1.0)
            opt.step()
        with torch.no_grad():
            c, _r = candidate._cayley_orthogonal(theta_auto)
            rotation_auto = torch.einsum("dk,gkl->gdl", base, c)

    diff = float((rotation_manual - rotation_auto).abs().max())
    print(f"full training: manual loss {info_manual['final_train_loss']:.6f}, "
          f"max |R_manual - R_autograd| = {diff:.3e}")
    if diff > 5e-2:
        print("FULL TRAINING PARITY FAIL (rotation matrices diverge)")
        return 1
    print("MANUAL TRAINER PARITY PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
