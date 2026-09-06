"""L1 direction sweep v3: fast-path STE T training across many real states.

Answers: does a per-state FlatQuant T=T1xT2 (trained 32-step STE Adam on the
fast legal codec) reduce the real quantized output error across layers/roles?
fast-path cost ~0.5s/state -> full 168-state scan is affordable (~2 min).

LOCAL diagnostic only.
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


def run_state(layer: int, role: str):
    w_dense = cache["weights"][layer][role].to(torch.float32)
    wq, ws = nvfp4_pair(w_dense)
    out_features, in_features = map(int, w_dense.shape)
    calib_pairs = []
    for sample in cache["calibration_activations"][role][:2]:
        act = sample[layer].to(torch.float32)
        calib_pairs.append(nvfp4_pair(act))
    out = sol.hif4_calibration_and_quantize_weight(wq, ws, calib_pairs)
    astate = out["activation_state"]
    weight = sol._dequantize_nvfp4_float32(wq, ws)
    d_inv = torch.ones(in_features, device=device)
    if astate.get("smooth_inv") is not None:
        d_inv = astate["smooth_inv"].to(device=device, dtype=torch.float32)
    perm = astate["permutation"]
    if perm is not None:
        perm = perm.to(device=device, dtype=torch.int64)
    else:
        perm = sol._identity_permutation(in_features, device)
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

    def enc(T, Td):
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

    T0 = torch.eye(_BLOCK, device=device)
    l0 = float(enc(T0, T0))
    B1 = torch.zeros(8, 8, device=device, requires_grad=True)
    B2 = torch.zeros(8, 8, device=device, requires_grad=True)
    opt = torch.optim.Adam([B1, B2], lr=0.01)
    for step in range(32):
        T1 = sym_zero_trace_exp(B1)
        T2 = sym_zero_trace_exp(B2)
        T = torch.kron(T1, T2)
        loss = enc(T, T)
        total = loss + 1e-3 * (B1.square().mean() + B2.square().mean())
        opt.zero_grad()
        total.backward()
        g = torch.cat([B1.grad.flatten(), B2.grad.flatten()])
        n = g.norm().clamp_min(1e-12)
        B1.grad.mul_(1.0 / n)
        B2.grad.mul_(1.0 / n)
        opt.step()
    lf = float(enc(torch.kron(sym_zero_trace_exp(B1), sym_zero_trace_exp(B2)),
                    torch.kron(sym_zero_trace_exp(B1), sym_zero_trace_exp(B2))).detach())
    return l0, lf


results = {}
t0 = time.perf_counter()
for layer in (0, 5, 11, 17, 23):
    for role in ROLES:
        try:
            l0, lf = run_state(layer, role)
        except Exception as exc:  # noqa: BLE001
            print(f"[L{layer}-{role}] ERROR {type(exc).__name__}: {exc}")
            continue
        delta = (lf - l0) / l0
        results[(layer, role)] = (l0, lf, delta)
        flag = "+" if delta < -0.01 else ("-" if delta > 0.01 else "=")
        print(f"[L{layer}-{role:8s}] {l0:.6f} -> {lf:.6f} rel {delta:+.3%} {flag}")
print("wall", round(time.perf_counter() - t0, 1), "s")
impr = sum(1 for d in results.values() if d[2] < -0.01)
degr = sum(1 for d in results.values() if d[2] > 0.01)
flat = sum(1 for d in results.values() if abs(d[2]) <= 0.01)
print(f"improved {impr} degraded {degr} flat {flat} of {len(results)}")