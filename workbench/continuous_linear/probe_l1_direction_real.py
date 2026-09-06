"""L1 direction probe v2: real data, fast-path STE training on ONE real state.

Uses the real cached Qwen weight + calibration activations (real NVFP4 input)
and the real L4 calibration state (smooth/perm/hadamard/rank/GPTQ) as the
deployment coordinate, then inserts a FlatQuant-style per-64-block T=T1xT2 and
trains it with 32-step STE Adam, measuring the real quantized output error.

Deployment-consistent fast path only (no GPTQ compensation, no h_inv rebuild):
this is a DIRECTION probe. If it shows no margin, the full-cost path would not
be justified; COST evidence (0.6-2.2s/step full hard forward) is recorded
separately in the mechanism card.

LOCAL diagnostic only; candidate stays a single file.
"""

from __future__ import annotations

import importlib.util
import math
import time
from pathlib import Path

import torch

CACHE = Path(r"d:\工作内容\AI竞赛\artifacts\official_eval\cache\qwen2.5-0.5b-proxy-v2.pt")
L4_PATH = Path(
    r"d:\工作内容\AI竞赛\solutions\v162_linear_l4-v189-linear-exact_officialNA_timeNA\solution.py"
)
spec = importlib.util.spec_from_file_location("l4sol", L4_PATH)
sol = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sol)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"device={device}")
_BLOCK = 64


def nvfp4_pair(dense: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    m = dense.reshape(*dense.shape[:-1], -1, 16)
    amax = m.abs().amax(dim=-1)
    scale = (amax / 6.0).clamp_min(1e-8)
    q = torch.round(m / scale.unsqueeze(-1)).clamp_(-6.0, 6.0)
    return q.flatten(-2, -1), scale


def apply_T_block(x: torch.Tensor, T: torch.Tensor) -> torch.Tensor:
    prefix = list(x.shape[:-1])
    c = x.shape[-1]
    y = x.reshape(*prefix, c // _BLOCK, _BLOCK)
    y = torch.matmul(y, T)
    return y.reshape(*prefix, c)


def sym_zero_trace_exp(B: torch.Tensor) -> torch.Tensor:
    B = (B + B.t()) / 2.0
    B = B - torch.eye(B.shape[0], device=B.device) * (B.diag().mean())
    if float(B.norm()) < 1e-12:
        return torch.matrix_exp(B)
    ev, evec = torch.linalg.eigh(B)
    log2_4 = math.log(2.0) / 4.0
    ev2 = ev.clamp(-log2_4, log2_4)
    B2 = (evec * ev2).mm(evec.t())
    return torch.matrix_exp(B2)


def probe_real_state(layer: int, role: str):
    cache = torch.load(CACHE, map_location="cpu", weights_only=False)
    w_dense = cache["weights"][layer][role].to(torch.float32)
    wq, ws = nvfp4_pair(w_dense)
    out_features, in_features = map(int, w_dense.shape)
    calib_samples = cache["calibration_activations"][role]
    # map: samples -> (layer encodable pairs); take first 2 windows
    calib_pairs = []
    for sample in calib_samples[:2]:
        act = sample[layer].to(torch.float32)
        calib_pairs.append(nvfp4_pair(act))
    print(
        f"[{role}-L{layer}] w {tuple(w_dense.shape)} calib pairs {len(calib_pairs)} "
        f"rows {[int(p[0].shape[0]) for p in calib_pairs]}"
    )

    out = sol.hif4_calibration_and_quantize_weight(wq, ws, calib_pairs)
    astate = out["activation_state"]
    weight = sol._dequantize_nvfp4_float32(wq, ws)
    d_inv = None
    if astate.get("smooth_inv") is not None:
        d_inv = astate["smooth_inv"].to(device=device, dtype=torch.float32)
    else:
        d_inv = torch.ones(in_features, device=device)
    perm = astate["permutation"]
    if perm is not None:
        perm = perm.to(device=device, dtype=torch.int64)
    else:
        perm = sol._identity_permutation(in_features, device)
    bs_size = int(astate.get("block_smooth_size", 0))
    bs_seed = int(astate.get("block_smooth_seed", 0))
    # NOTE: L4 restored static-actorder variant keeps block_smooth_size=0 path
    # for GPTQ; use the state values as-is so the probe stays deployment-aligned.
    weight_smooth = sol._linear_pair_transform(
        weight.to(device), d_inv.reciprocal(), perm, bs_size, bs_seed, weight_side=True
    )
    x_list = []
    for (xq, xs) in calib_pairs:
        x = sol._dequantize_nvfp4_float32(xq, xs).to(device)
        x = x * d_inv.unsqueeze(0)
        x = x.index_select(-1, perm)
        x_list.append(x)
    x_s = torch.cat(x_list, dim=0)  # [rows, in]
    weight_smooth = weight_smooth.to(device)
    ref = (x_s.detach() @ weight_smooth.detach().t()).detach()
    std_mse = (ref - ref.mean()).square().mean().clamp_min(1e-12)
    offs = (-1, 1, 2, 3)
    et, am = 0.0, 0.0

    def encode_loss(T):
        T_inv = torch.linalg.inv(T).t()
        W_T = apply_T_block(weight_smooth, T_inv)
        X_T = apply_T_block(x_s, T)
        W_T_d = W_T.detach()
        X_T_d = X_T.detach()
        wp = sol._dense_to_hif4(
            W_T_d,
            importance=None,
            group_gram=None,
            search_offsets=offs,
            error_threshold=et,
            accept_margin=am,
            max_refine_ratio=0.0,
            max_refine_blocks=0,
        )
        wh = sol._dequantize_hif4(wp).to(torch.float32)
        ap = sol._dense_to_hif4(
            X_T_d,
            importance=None,
            group_gram=None,
            search_offsets=offs,
            error_threshold=et,
            accept_margin=am,
            max_refine_ratio=0.0,
            max_refine_blocks=0,
        )
        ah = sol._dequantize_hif4(ap).to(torch.float32)
        wh_ste = wh + (W_T - W_T_d)
        ah_ste = ah + (X_T - X_T_d)
        out_t = ah_ste @ wh_ste.t()
        return ((out_t - ref).square().mean() / std_mse, T)

    T0 = torch.eye(_BLOCK, device=device)
    l0, _ = encode_loss(T0)
    print(f"  [{role}-L{layer}] baseline(T=I) norm loss = {float(l0):.6f}")

    # sanitize offs/tolerance to match dense_to_hif4 signature expectations
    offs = (1,)
    def encode_loss_simple(T):
        T_inv = torch.linalg.inv(T).t()
        W_T = apply_T_block(weight_smooth, T_inv)
        X_T = apply_T_block(x_s, T)
        W_T_d = W_T.detach()
        X_T_d = X_T.detach()
        wp = sol._dense_to_hif4(W_T_d, importance=None, group_gram=None,
                                search_offsets=offs, error_threshold=0.0,
                                accept_margin=0.0, max_refine_ratio=0.0,
                                max_refine_blocks=0)
        wh = sol._dequantize_hif4(wp).to(torch.float32)
        ap = sol._dense_to_hif4(X_T_d, importance=None, group_gram=None,
                                search_offsets=offs, error_threshold=0.0,
                                accept_margin=0.0, max_refine_ratio=0.0,
                                max_refine_blocks=0)
        ah = sol._dequantize_hif4(ap).to(torch.float32)
        wh_ste = wh + (W_T - W_T_d)
        ah_ste = ah + (X_T - X_T_d)
        out_t = ah_ste @ wh_ste.t()
        return (out_t - ref).square().mean() / std_mse

    l0s = encode_loss_simple(T0)
    print(f"  [{role}-L{layer}] baseline(offs=1) norm loss = {float(l0s):.6f}")

    B1 = torch.zeros(8, 8, device=device, requires_grad=True)
    B2 = torch.zeros(8, 8, device=device, requires_grad=True)
    opt = torch.optim.Adam([B1, B2], lr=0.01)
    reg = 1e-3
    t0 = time.perf_counter()
    for step in range(32):
        T1 = sym_zero_trace_exp(B1)
        T2 = sym_zero_trace_exp(B2)
        T = torch.kron(T1, T2)
        loss = encode_loss_simple(T)
        total = loss + reg * (B1.square().mean() + B2.square().mean())
        opt.zero_grad()
        total.backward()
        g = torch.cat([B1.grad.flatten(), B2.grad.flatten()])
        norm = g.norm().clamp_min(1e-12)
        B1.grad.mul_(1.0 / norm)
        B2.grad.mul_(1.0 / norm)
        opt.step()
        if step in (0, 7, 15, 31):
            print(f"    step {step + 1}: norm loss = {float(loss.detach()):.6f}")
    dt = time.perf_counter() - t0
    print(f"  [{role}-L{layer}] 32-step STE wall {dt:.1f}s ({dt / 32:.3f}s/step)")


probe_real_state(0, "o")
probe_real_state(0, "fc_up")
probe_real_state(0, "proj")