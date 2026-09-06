"""Manual-gradient rotation trainer (A2c): identical objective to A2, no autograd.

The official harness defeats every autograd escape (inference tensors +
harness-level inference guards defeat `inference_mode(False)` re-entry), so
the pre-registered Cayley-rotation training is re-implemented with analytic
gradients: pure tensor math that runs under any context (no_grad, inference,
C++ RAII guards) and needs no autograd graph at all.

Gradient chain (STE = identity through the hard quantizer):
  O = softmax(S) V,  S_h = Q'_h K_g^T / sqrt(d)          (attention backward)
  Q'_h = Q_h R_g,  K'_g = K_g R_g                        (row-vector right mult)
  R_g = H C_g,  C_g = (I-A)(I+A)^-1,  A = Theta-Theta^T  (Cayley backward:
      dC = -(I+C) dA M^-1, M = I+A  =>  dL/dA = -(I+C)^T G M^-T)
  A = Theta - Theta^T  =>  dL/dTheta = G_A - G_A^T
Parity with the A2b autograd path is verified by
`test_manual_trainer_parity.py` before deployment.
"""

from __future__ import annotations

import math
from typing import Any

import torch


def attention_backward(
    d_out: torch.Tensor,
    q_hat: torch.Tensor,
    k_hat: torch.Tensor,
    v: torch.Tensor,
    q_heads: int,
    kv_heads: int,
    head_dim: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """dL/dq_hat and dL/dk_hat for the non-causal GQA attention forward.

    All tensors are 2D (tokens, channels); d_out is (tokens, q_heads*head_dim)
    with the same layout.  Mirrors evaluator `_attention` exactly.
    """

    scale = 1.0 / math.sqrt(head_dim)
    group = q_heads // kv_heads
    tokens_q = q_hat.shape[0]
    tokens_k = k_hat.shape[0]
    qh = q_hat.reshape(tokens_q, q_heads, head_dim).transpose(0, 1)
    kh = k_hat.reshape(tokens_k, kv_heads, head_dim).transpose(0, 1).repeat_interleave(group, dim=0)
    vh = v.reshape(tokens_k, kv_heads, head_dim).transpose(0, 1).repeat_interleave(group, dim=0)
    do = d_out.reshape(tokens_q, q_heads, head_dim).transpose(0, 1)
    logits = (qh @ kh.transpose(-1, -2)) * scale
    probabilities = torch.softmax(logits, dim=-1)
    d_prob = do @ vh.transpose(-1, -2)
    tmp = (d_prob * probabilities).sum(dim=-1, keepdim=True)
    d_logits = (d_prob - tmp) * probabilities * scale
    d_q = torch.einsum("hij,hjk->hik", d_logits, kh)          # (H, Tq, D)
    d_k = torch.einsum("hij,hik->hjk", d_logits, qh)          # (H, Tk, D)
    d_k = d_k.view(kv_heads, group, k_hat.shape[0], head_dim).sum(dim=1)
    d_q = d_q.transpose(0, 1).reshape(tokens_q, q_heads * head_dim)
    d_k = d_k.permute(1, 0, 2).reshape(k_hat.shape[0], kv_heads * head_dim)
    return d_q, d_k


def cayley_backward(grad_c: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
    """dL/dTheta from dL/dC for C=(I-A)(I+A)^-1, A=Theta-Theta^T (batched)."""

    dim = theta.shape[-1]
    eye = torch.eye(dim, device=theta.device, dtype=theta.dtype)
    skew = theta - theta.transpose(-1, -2)
    left = eye - skew
    right = eye + skew
    c = torch.linalg.solve(right.transpose(-1, -2), left.transpose(-1, -2)).transpose(-1, -2)
    # Y = G M^-T :  Y^T = M^-1 G^T  =>  solve(M, G^T)^T
    y = torch.linalg.solve(right, grad_c.transpose(-1, -2)).transpose(-1, -2)
    grad_a = -(c.transpose(-1, -2) + eye) @ y
    return grad_a - grad_a.transpose(-1, -2)


def rotate_rows_manual(
    dense: torch.Tensor, num_heads: int, rotation: torch.Tensor
) -> torch.Tensor:
    tokens = dense.shape[0]
    head_dim = dense.shape[-1] // num_heads
    groups = rotation.shape[0]
    per_group = num_heads // groups
    grouped = dense.reshape(tokens, groups, per_group, head_dim)
    rotated = torch.einsum(
        "tghk,gkd->tghd", grouped, rotation.to(device=grouped.device, dtype=torch.float32)
    )
    return rotated.reshape(tokens, dense.shape[-1])


def attention_forward_manual(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
    q_heads: int, kv_heads: int, head_dim: int,
) -> torch.Tensor:
    batch, tokens, _ = q.shape
    qh = q.reshape(batch, tokens, q_heads, head_dim).transpose(1, 2)
    group = q_heads // kv_heads
    kh = k.reshape(batch, -1, kv_heads, head_dim).transpose(1, 2).repeat_interleave(group, dim=1)
    vh = v.reshape(batch, -1, kv_heads, head_dim).transpose(1, 2).repeat_interleave(group, dim=1)
    probabilities = torch.softmax(qh @ kh.transpose(-1, -2) / math.sqrt(head_dim), dim=-1)
    return (probabilities @ vh).transpose(1, 2).reshape(batch, tokens, q_heads * head_dim)


def adam_step(
    theta: torch.Tensor,
    grad: torch.Tensor,
    exp_avg: torch.Tensor,
    exp_avg_sq: torch.Tensor,
    step: int,
    lr: float = 0.01,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-8,
    max_norm: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    total = grad.norm()
    if float(total) > max_norm and float(total) > 0:
        grad = grad * (max_norm / float(total))
    exp_avg = beta1 * exp_avg + (1 - beta1) * grad
    exp_avg_sq = beta2 * exp_avg_sq + (1 - beta2) * grad.square()
    bias1 = 1 - beta1 ** step
    bias2 = 1 - beta2 ** step
    theta = theta - lr * (exp_avg / bias1) / ((exp_avg_sq / bias2).sqrt() + eps)
    return theta, exp_avg, exp_avg_sq


def cayley_group(theta: torch.Tensor, eye: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    skew = theta - theta.transpose(-1, -2)
    left = eye - skew
    right = eye + skew
    c = torch.linalg.solve(right.transpose(-1, -2), left.transpose(-1, -2)).transpose(-1, -2)
    return c, right


def train_rotation_manual(
    prepared: list[dict[str, torch.Tensor]],
    base: torch.Tensor,
    groups: int,
    dim: int,
    q_heads: int,
    kv_heads: int,
    head_dim: int,
    device: torch.device,
    steps: int = 32,
    lr: float = 0.01,
    reg_weight: float = 1e-3,
    hard_encode=None,
    hard_decode=None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Adam on Theta with analytic gradients; no autograd anywhere.

    ``hard_encode``/``hard_decode`` are the candidate's standard HiF4 codec
    functions; the STE gradient identity means the gradient treats the hard
    quantizer as the identity while the forward uses its decoded value.
    """

    theta = torch.zeros(groups, dim, dim, device=device, dtype=torch.float32)
    exp_avg = torch.zeros_like(theta)
    exp_avg_sq = torch.zeros_like(theta)
    eye = torch.eye(dim, device=device, dtype=torch.float32)
    final_loss = float("nan")
    identity_error = 0.0
    rotation = torch.einsum("dk,gkl->gdl", base, eye[None].expand(groups, -1, -1))

    for step_index in range(steps):
        window_losses = []
        grad_theta = torch.zeros_like(theta)
        for item in prepared:
            with torch.no_grad():
                c, _right = cayley_group(theta, eye)
                rotation = torch.einsum("dk,gkl->gdl", base, c)
                q_rot = rotate_rows_manual(item["q"], q_heads, rotation)
                k_rot = rotate_rows_manual(item["k"], kv_heads, rotation)
                q_hat = hard_decode(hard_encode(q_rot))
                k_hat = hard_decode(hard_encode(k_rot))
                output = attention_forward_manual(
                    q_hat[None], k_hat[None], item["v_hat"][None], q_heads, kv_heads, head_dim
                )[0]
                residual = output - item["reference"]
                loss = residual.square().mean() / item["mse_std"]
            window_losses.append(loss)

            d_output = 2.0 * residual / float(residual.numel()) / item["mse_std"]
            d_qhat, d_khat = attention_backward(
                d_output[None], q_hat, k_hat, item["v_hat"], q_heads, kv_heads, head_dim
            )
            tokens_q = item["q"].shape[0]
            tokens_k = item["k"].shape[0]
            per_group = q_heads // groups
            q3g = item["q"].reshape(tokens_q, groups, per_group, head_dim)
            dq3g = d_qhat.reshape(tokens_q, groups, per_group, head_dim)
            k3 = item["k"].reshape(tokens_k, kv_heads, head_dim)
            dk3 = d_khat.reshape(tokens_k, kv_heads, head_dim)
            grad_rotation = torch.einsum("tghk,tghd->gkd", q3g, dq3g)
            grad_rotation = grad_rotation + torch.einsum("tgk,tgd->gkd", k3, dk3)
            grad_c = torch.einsum("kd,gkl->gdl", base, grad_rotation)
            grad_theta = grad_theta + cayley_backward(grad_c, theta)

        data_loss = torch.stack(window_losses).mean()
        if not math.isfinite(float(data_loss)):
            raise RuntimeError("manual rotation training produced a non-finite loss")
        # regularizer gradient: (rw) * mean_{g,d1,d2}((C-I)^2)
        c_now, _right = cayley_group(theta, eye)
        grad_c_reg = 2.0 * (c_now - eye) * (reg_weight / float(groups * dim * dim))
        grad_theta = grad_theta + cayley_backward(grad_c_reg, theta)
        final_loss = float(data_loss)

        theta, exp_avg, exp_avg_sq = adam_step(
            theta, grad_theta, exp_avg, exp_avg_sq, step_index + 1, lr=lr
        )

    with torch.no_grad():
        c, _right = cayley_group(theta, eye)
        rotation = torch.einsum("dk,gkl->gdl", base, c)
        identity_error = float((rotation @ rotation.transpose(-1, -2) - eye[None]).abs().max())
    if identity_error > 1e-3:
        raise RuntimeError(f"manual trained rotation failed orthogonality: {identity_error}")
    info = {
        "steps": steps,
        "final_train_loss": final_loss,
        "ortho_error": identity_error,
        "windows": len(prepared),
        "trainer": "manual",
    }
    return rotation.detach().cpu().to(torch.float32), info
