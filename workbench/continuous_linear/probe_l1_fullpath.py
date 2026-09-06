"""L1 direction probe v4: FULL deployment path (weight GPTQ + activation GPTQ
with h_inv/gram/importance/offsets rebuilt in the T-transformed coordinate)
evaluates whether a fast-path-trained T reduces the real output error.

This is the decisive check: fast-path STE may mislead. If T also fails on the
full deployment path, L1 (FlatQuant Kronecker rotation) is LOCAL_NEGATIVE.

LOCAL diagnostic only; ~1 full hard forward per eval (~1-2s).
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
_BLOCK = 64
ROLES = ("q", "k", "v", "o", "fc_gate", "fc_up", "proj")
cache = torch.load(CACHE, map_location="cpu", weights_only=False)


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


def train_T_fast(weight_smooth, x_s, ref, std_mse, steps=32):
    def enc(T):
        T_inv = torch.linalg.inv(T).t()
        W_T = apply_T_block(weight_smooth, T_inv)
        X_T = apply_T_block(x_s, T)
        W_T_d = W_T.detach()
        X_T_d = X_T.detach()
        wp = sol._dense_to_hif4(W_T_d, importance=None, group_gram=None,
                                search_offsets=(1,), error_threshold=0.0,
                                accept_margin=0.0, max_refine_ratio=0.0,
                                max_refine_blocks=0)
        wh = sol._dequantize_hif4(wp).to(torch.float32)
        ap = sol._dense_to_hif4(X_T_d, importance=None, group_gram=None,
                                search_offsets=(1,), error_threshold=0.0,
                                accept_margin=0.0, max_refine_ratio=0.0,
                                max_refine_blocks=0)
        ah = sol._dequantize_hif4(ap).to(torch.float32)
        wh_ste = wh + (W_T - W_T_d)
        ah_ste = ah + (X_T - X_T_d)
        return ((ah_ste @ wh_ste.t() - ref).square().mean() / std_mse)

    B1 = torch.zeros(8, 8, device=device, requires_grad=True)
    B2 = torch.zeros(8, 8, device=device, requires_grad=True)
    opt = torch.optim.Adam([B1, B2], lr=0.01)
    for _ in range(steps):
        T1 = sym_zero_trace_exp(B1)
        T2 = sym_zero_trace_exp(B2)
        T = torch.kron(T1, T2)
        loss = enc(T)
        total = loss + 1e-3 * (B1.square().mean() + B2.square().mean())
        opt.zero_grad()
        total.backward()
        g = torch.cat([B1.grad.flatten(), B2.grad.flatten()])
        n = g.norm().clamp_min(1e-12)
        B1.grad.mul_(1.0 / n)
        B2.grad.mul_(1.0 / n)
        opt.step()
    return torch.kron(sym_zero_trace_exp(B1), sym_zero_trace_exp(B2)).detach()


def full_path_loss(weight, x_s, astate, T):
    """Full deployment path with T folded in: weight GPTQ in T^{-T} coord,
    activation GPTQ in T coord, output error normalized by std MSE."""
    T_inv = torch.linalg.inv(T).t()
    W_T = apply_T_block(weight, T_inv)
    X_T = apply_T_block(x_s, T)
    blocks = int(W_T.shape[1]) // _BLOCK
    # rebuild transformed-coordinate gram (X_T^T X_T) and importance
    gram_full = X_T.detach().t() @ X_T.detach() / max(float(X_T.shape[0]), 1e-9)
    reg = 0.2
    H = gram_full.clone()
    H.diagonal().add_(reg * float(H.diagonal().mean()))
    try:
        L = torch.linalg.cholesky(H)
        H_inv = torch.cholesky_inverse(L)
    except RuntimeError:
        H.diagonal().add_(reg * float(H.diagonal().mean()) * 10.0)
        try:
            L = torch.linalg.cholesky(H)
            H_inv = torch.cholesky_inverse(L)
        except RuntimeError:
            return None
    group_gram = None
    g = gram_full.clone()
    wgram = g.reshape(blocks, 8, 2, 4, 4).unsqueeze(0).expand(
        int(W_T.shape[0]), blocks, 8, 2, 4, 4
    ).contiguous()
    wp = sol._gptq_quantize_weight(
        W_T,
        H,
        importance=x_s.detach().square().mean(dim=0).clamp_min(1e-9),
        group_gram=wgram,
        search_offsets=tuple(int(o) for o in astate["offsets"].tolist()),
        error_threshold=0.0,
        accept_margin=0.0,
        max_refine_ratio=0.0,
        max_refine_blocks=0,
        full_sweep_top_k=0,
        regularization=reg,
    )
    wh = sol._dequantize_hif4(wp).to(torch.float32)
    wout_gram = wh.t() @ wh
    H_act = wout_gram.clone()
    reg_a = 0.2
    H_act.diagonal().add_(reg_a * float(H_act.diagonal().mean()))
    try:
        L_a = torch.linalg.cholesky(H_act)
        H_inv_a = torch.cholesky_inverse(L_a)
    except RuntimeError:
        H_act.diagonal().add_(reg_a * float(H_act.diagonal().mean()) * 10.0)
        L_a = torch.linalg.cholesky(H_act)
        H_inv_a = torch.cholesky_inverse(L_a)
    a_gram = wout_gram.reshape(blocks, 8, 2, 4, 4).unsqueeze(0).expand(
        int(X_T.shape[0]), blocks, 8, 2, 4, 4
    ).contiguous()
    ap = sol._activation_gptq_quantize(
        X_T,
        H_inv_a,
        importance=X_T.detach().square().mean(dim=0).clamp_min(1e-9),
        group_gram=a_gram,
        search_offsets=tuple(int(o) for o in astate["offsets"].tolist()),
        error_threshold=0.0,
        accept_margin=0.0,
        max_refine_ratio=0.0,
        max_refine_blocks=0,
    )
    ah = sol._dequantize_hif4(ap).to(torch.float32)
    out = ah @ wh.t()
    ref = x_s @ (weight.detach().t())
    std_mse = (ref - ref.mean()).square().mean().clamp_min(1e-12)
    return float(((out - ref).square().mean() / std_mse))


def run_state(layer: int, role: str):
    w_dense = cache["weights"][layer][role].to(torch.float32)
    wq, ws = nvfp4_pair(w_dense)
    calib_pairs = []
    for sample in cache["calibration_activations"][role][:2]:
        act = sample[layer].to(torch.float32)
        calib_pairs.append(nvfp4_pair(act))
    out = sol.hif4_calibration_and_quantize_weight(wq, ws, calib_pairs)
    astate = out["activation_state"]
    weight = sol._dequantize_nvfp4_float32(wq, ws)
    d_inv = torch.ones(weight.shape[1], device=device)
    if astate.get("smooth_inv") is not None:
        d_inv = astate["smooth_inv"].to(device=device, dtype=torch.float32)
    perm = astate["permutation"]
    if perm is not None:
        perm = perm.to(device=device, dtype=torch.int64)
    else:
        perm = sol._identity_permutation(weight.shape[1], device)
    bs_size = int(astate.get("block_smooth_size", 0))
    bs_seed = int(astate.get("block_smooth_seed", 0))
    weight_smooth = sol._linear_pair_transform(
        weight.to(device), d_inv.reciprocal(), perm, bs_size, bs_seed, weight_side=True
    )
    x_list = []
    for (xq, xs) in calib_pairs:
        x = sol._dequantize_nvfp4_float32(xq, xs).to(device)
        x = x * d_inv.unsqueeze(0)
        x = x.index_select(-1, perm)
        x_list.append(x)
    x_s = torch.cat(x_list, dim=0)
    ref = (x_s.detach() @ weight_smooth.detach().t()).detach()
    std_mse = (ref - ref.mean()).square().mean().clamp_min(1e-12)

    T0 = torch.eye(_BLOCK, device=device)
    l0 = full_path_loss(weight_smooth, x_s, astate, T0)
    T = train_T_fast(weight_smooth, x_s, ref, std_mse)
    lf = full_path_loss(weight_smooth, x_s, astate, T)
    return l0, lf


results = {}
for layer, role in ((0, "o"), (0, "fc_up"), (11, "proj"), (17, "o"), (23, "fc_up")):
    try:
        l0, lf = run_state(layer, role)
    except Exception as exc:  # noqa: BLE001
        print(f"[L{layer}-{role}] ERROR {type(exc).__name__}: {exc}")
        continue
    if l0 is None or lf is None:
        print(f"[L{layer}-{role}] full-path unavailable")
        continue
    delta = (lf - l0) / l0
    flag = "+" if delta < -0.01 else ("-" if delta > 0.01 else "=")
    print(f"[L{layer}-{role:8s}] full-path {l0:.6f} -> {lf:.6f} rel {delta:+.3%} {flag}")
    results[(layer, role)] = (l0, lf, delta)