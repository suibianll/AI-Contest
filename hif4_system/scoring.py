from __future__ import annotations

import math

import torch


def competition_score(
    reference: torch.Tensor,
    standard: torch.Tensor,
    player: torch.Tensor,
    epsilon: float = 1.0e-30,
) -> float:
    if reference.shape != standard.shape or reference.shape != player.shape:
        raise ValueError("reference, standard, and player shapes must match")
    reference_f = reference.to(torch.float32)
    mse_std = torch.mean((standard.to(torch.float32) - reference_f).square())
    mse_player = torch.mean((player.to(torch.float32) - reference_f).square())
    return float((mse_std - mse_player) / torch.clamp_min(mse_std, epsilon))


def attention_output(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q_num_heads: int,
    kv_num_heads: int,
    head_dim: int,
    causal: bool,
) -> torch.Tensor:
    if q.ndim != 2 or k.ndim != 2 or v.ndim != 2:
        raise ValueError("attention inputs must be rank-2 [seq, channels]")
    q_len, q_channels = q.shape
    k_len, k_channels = k.shape
    if v.shape != k.shape or q_channels != q_num_heads * head_dim:
        raise ValueError("attention channel shapes do not match head metadata")
    if k_channels != kv_num_heads * head_dim or q_num_heads % kv_num_heads != 0:
        raise ValueError("invalid MHA/GQA head metadata")
    if causal and q_len != k_len:
        raise ValueError("causal attention requires equal query/key lengths")
    qh = q.to(torch.float32).reshape(q_len, q_num_heads, head_dim).transpose(0, 1)
    kh = k.to(torch.float32).reshape(k_len, kv_num_heads, head_dim).transpose(0, 1)
    vh = v.to(torch.float32).reshape(k_len, kv_num_heads, head_dim).transpose(0, 1)
    repeat = q_num_heads // kv_num_heads
    kh = kh.repeat_interleave(repeat, dim=0)
    vh = vh.repeat_interleave(repeat, dim=0)
    logits = torch.matmul(qh, kh.transpose(-1, -2)) / math.sqrt(float(head_dim))
    if causal:
        mask = torch.ones((q_len, k_len), dtype=torch.bool, device=logits.device).triu(1)
        logits = logits.masked_fill(mask.unsqueeze(0), float("-inf"))
    probabilities = torch.softmax(logits, dim=-1)
    output = torch.matmul(probabilities, vh)
    return output.transpose(0, 1).reshape(q_len, q_num_heads * head_dim)
