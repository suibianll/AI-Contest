"""Build candidate_c (A2c): candidate_b with the manual-gradient trainer.

A2b officially scored exactly the v162 zero point because the official
harness defeats every autograd escape, so the training fallback engaged.
A2c replaces the autograd trainer with the analytic-gradient trainer
(verified element-wise against autograd; gate loss on real layer-15 data
0.494 vs autograd 0.554, identity = 1.0).  All `inference_mode(False)`
usage is removed: the manual trainer is pure tensor math and runs under
any harness context.  Every guard from A2b is kept.
"""

from __future__ import annotations

from pathlib import Path

ATT = Path(__file__).resolve().parent
SRC = ATT / "candidate_b/solution.py"
OUT = ATT / "candidate_c/solution.py"

MANUAL = '''

def _m_attention_backward(
    d_out: torch.Tensor,
    q_hat: torch.Tensor,
    k_hat: torch.Tensor,
    v: torch.Tensor,
    q_heads: int,
    kv_heads: int,
    head_dim: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """dL/dq_hat and dL/dk_hat for the non-causal GQA attention forward."""

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
    d_q = torch.einsum("hij,hjk->hik", d_logits, kh)
    d_k = torch.einsum("hij,hik->hjk", d_logits, qh)
    d_k = d_k.view(kv_heads, group, tokens_k, head_dim).sum(dim=1)
    d_q = d_q.transpose(0, 1).reshape(tokens_q, q_heads * head_dim)
    d_k = d_k.permute(1, 0, 2).reshape(tokens_k, kv_heads * head_dim)
    return d_q, d_k


def _m_cayley_backward(grad_c: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
    dim = theta.shape[-1]
    eye = torch.eye(dim, device=theta.device, dtype=theta.dtype)
    skew = theta - theta.transpose(-1, -2)
    left = eye - skew
    right = eye + skew
    c = torch.linalg.solve(right.transpose(-1, -2), left.transpose(-1, -2)).transpose(-1, -2)
    y = torch.linalg.solve(right, grad_c.transpose(-1, -2)).transpose(-1, -2)
    grad_a = -(c.transpose(-1, -2) + eye) @ y
    return grad_a - grad_a.transpose(-1, -2)


def _m_train_rotation(
    prepared: list[dict[str, torch.Tensor]],
    base: torch.Tensor,
    groups: int,
    dim: int,
    q_heads: int,
    kv_heads: int,
    head_dim: int,
    device: torch.device,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Adam on Theta with analytic gradients; no autograd anywhere."""

    theta = torch.zeros(groups, dim, dim, device=device, dtype=torch.float32)
    exp_avg = torch.zeros_like(theta)
    exp_avg_sq = torch.zeros_like(theta)
    eye = torch.eye(dim, device=device, dtype=torch.float32)
    final_loss = float("nan")
    identity_error = 0.0

    for step_index in range(_TRAIN_STEPS):
        window_losses = []
        grad_theta = torch.zeros_like(theta)
        for item in prepared:
            skew = theta - theta.transpose(-1, -2)
            left = eye - skew
            right = eye + skew
            c = torch.linalg.solve(right.transpose(-1, -2), left.transpose(-1, -2)).transpose(-1, -2)
            rotation = torch.einsum("dk,gkl->gdl", base, c)
            q_rot = _rotate_rows(item["q"], q_heads, rotation)
            k_rot = _rotate_rows(item["k"], kv_heads, rotation)
            q_hat = _decode_hif5_fields(_encode_standard_hif4(q_rot))
            k_hat = _decode_hif5_fields(_encode_standard_hif4(k_rot))
            output = _attention_forward(
                q_hat[None], k_hat[None], item["v_hat"][None], q_heads, kv_heads, head_dim
            )[0]
            residual = output - item["reference"]
            loss = residual.square().mean() / item["mse_std"]
            window_losses.append(loss)

            d_output = 2.0 * residual / float(residual.numel()) / item["mse_std"]
            d_qhat, d_khat = _m_attention_backward(
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
            grad_theta = grad_theta + _m_cayley_backward(grad_c, theta)

        data_loss = torch.stack(window_losses).mean()
        if not math.isfinite(float(data_loss)):
            raise RuntimeError("manual rotation training produced a non-finite loss")
        c_now, _right = _m_cayley_pair(theta)
        grad_c_reg = 2.0 * (c_now - eye) * (_REG_WEIGHT / float(groups * dim * dim))
        grad_theta = grad_theta + _m_cayley_backward(grad_c_reg, theta)
        final_loss = float(data_loss)

        total = grad_theta.norm()
        if float(total) > _TRAIN_CLIP and float(total) > 0:
            grad_theta = grad_theta * (_TRAIN_CLIP / float(total))
        exp_avg = 0.9 * exp_avg + 0.1 * grad_theta
        exp_avg_sq = 0.999 * exp_avg_sq + 0.001 * grad_theta.square()
        bias1 = 1 - 0.9 ** (step_index + 1)
        bias2 = 1 - 0.999 ** (step_index + 1)
        theta = theta - _TRAIN_LR * (exp_avg / bias1) / ((exp_avg_sq / bias2).sqrt() + 1e-8)

    skew = theta - theta.transpose(-1, -2)
    left = eye - skew
    right = eye + skew
    c = torch.linalg.solve(right.transpose(-1, -2), left.transpose(-1, -2)).transpose(-1, -2)
    rotation = torch.einsum("dk,gkl->gdl", base, c)
    identity_error = float((rotation @ rotation.transpose(-1, -2) - eye[None]).abs().max())
    if identity_error > _ORTHO_TOLERANCE:
        raise RuntimeError(f"manual trained rotation failed orthogonality: {identity_error}")
    info = {
        "steps": _TRAIN_STEPS,
        "final_train_loss": final_loss,
        "ortho_error": identity_error,
        "windows": len(prepared),
        "trainer": "manual",
    }
    return rotation.detach().cpu().to(torch.float32), info


def _m_cayley_pair(theta: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    dim = theta.shape[-1]
    eye = torch.eye(dim, device=theta.device, dtype=theta.dtype)
    skew = theta - theta.transpose(-1, -2)
    left = eye - skew
    right = eye + skew
    c = torch.linalg.solve(right.transpose(-1, -2), left.transpose(-1, -2)).transpose(-1, -2)
    return c, right
'''

src = SRC.read_text(encoding="utf-8")

# 1. neutralize inference_mode(False) in _normal_tensor (no autograd needed)
old_normal = '''def _normal_tensor(t: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Rebuild a tensor as a normal (non-inference) tensor on ``device``."""

    with torch.inference_mode(False):
        normal = t.detach().to(device=device, dtype=torch.float32).clone()
    if normal.is_inference():
        plain = torch.empty(
            tuple(normal.shape), device=device, dtype=torch.float32
        )
        plain.copy_(normal)
        normal = plain
    return normal'''
new_normal = '''def _normal_tensor(t: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Plain float32 copy on ``device``; the manual trainer needs no autograd."""

    return t.detach().to(device=device, dtype=torch.float32).clone()'''
assert src.count(old_normal) == 1
src = src.replace(old_normal, new_normal)

# 2. replace the autograd training with the manual trainer
old_call = """    with torch.inference_mode(False), torch.enable_grad():
        return _train_rotation_loop(
            prepared, base, groups, dim, q_heads, kv_heads, head_dim, device
        )"""
new_call = """    return _m_train_rotation(
        prepared, base, groups, dim, q_heads, kv_heads, head_dim, device
    )"""
assert src.count(old_call) == 1
src = src.replace(old_call, new_call)

# 3. drop the now-dead _train_rotation_loop body (between its def and the
#    next top-level def) and append the manual helpers
loop_start = src.find("def _train_rotation_loop(")
assert loop_start != -1
next_def = src.find("\ndef ", loop_start + 1)
assert next_def != -1
src = src[:loop_start] + src[next_def + 1:]
# the removed body left the return of _train_rotation dangling? no: the call
# above replaced the only use; _train_rotation ends right after it.
anchor = "def _calibration_attention_impl("
assert src.count(anchor) == 1
src = src.replace(anchor, MANUAL.strip() + "\n\n\n" + anchor)

OUT.write_text(src, encoding="utf-8")
print("candidate_c (A2c) written")
