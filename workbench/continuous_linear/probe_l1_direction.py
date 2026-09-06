"""L1 direction probe: does a FlatQuant-style structured invertible transform
T = T1 x T2 (8x8 matrix-exponential factors, Kronecker) reduce the real
quantized output error on a clean deployment coordinate?

Design (workpackage continuous-linear.md L1):
- One T per weight state, shared across all 64-blocks.
- Ti = exp(Bi), Bi symmetric zero-trace, eigenvalues clipped to
  [-log(2)/4, +log(2)/4] -> cond(T) <= 2; start from I.
- Deployment coordinates: per-64-block X right-multiplies T, W right-multiplies
  T^{-T}; inverse computed only during calibration.
- 32 steps Adam lr=0.01, grad-norm 1, reg 1e-3*mean(B^2).
- STE round-trip through the real codecs; loss is true output error,
  normalized by the same-fold standard output MSE.

This is a LOCAL diagnostic only (single state, clean coords, rank/offsets
fixed). It answers whether the mechanism has local margin BEFORE committing to
the full implementation, and it records the per-step full-hard-forward cost so
the mechanism card can decide COST/DESIGN_HOLD per the workpackage.
"""

from __future__ import annotations

import importlib.util
import math
import time
from pathlib import Path

import torch

L4_PATH = Path(
    r"d:\工作内容\AI竞赛\solutions\v162_linear_l4-v189-linear-exact_officialNA_timeNA\solution.py"
)
spec = importlib.util.spec_from_file_location("l4sol", L4_PATH)
sol = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sol)

torch.manual_seed(0)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"device={device} torch={torch.__version__}")
_BLOCK = 64


def nvfp4_approx(dense: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    m = dense.reshape(*dense.shape[:-1], -1, 16)
    amax = m.abs().amax(dim=-1)
    scale = (amax / 6.0).clamp_min(1e-8)
    q = torch.round(m / scale.unsqueeze(-1)).clamp_(-6.0, 6.0)
    return q.flatten(-2, -1), scale


def build_clean_state(in_features: int, out_features: int):
    w_dense = (torch.randn(out_features, in_features, device=device) * 0.02).to(
        torch.float32
    )
    wq, ws = nvfp4_approx(w_dense)
    calib = []
    for _ in range(2):
        x_dense = torch.randn(128, in_features, device=device) * 0.5
        xq, xs = nvfp4_approx(x_dense)
        calib.append((xq, xs))

    # First run the real L4 calibration to get the selected smooth/perm/hadamard
    # deployment coordinate, then RESET gram/h_inv/importance so the probe works
    # on a clean coordinate without rank/offset coupling.
    out = sol.hif4_calibration_and_quantize_weight(wq, ws, calib)
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
    import itertools

    bsm = 0
    bss = 0
    for (s, sd) in itertools.product((0,), (0,)):
        bsm, bss = s, sd
    weight_smooth = sol._linear_pair_transform(
        weight, d_inv.reciprocal(), perm, bsm, bss, weight_side=True
    )
    x_s = []
    for (xq, xs) in calib:
        x = sol._dequantize_nvfp4_float32(xq, xs)
        t = (x * d_inv.unsqueeze(0)).index_select(-1, perm)
        x_s.append(t)
    x_s = torch.cat(x_s, dim=0)  # [256, in]
    xw = x_s.detach() @ weight_smooth.detach().t()
    std_mse = (xw - xw.mean()).square().mean().clamp_min(1e-12)
    ref = x_s.detach() @ weight_smooth.detach().t()  # continuous reference
    return weight_smooth, x_s, ref, std_mse


def apply_T_block(x: torch.Tensor, T: torch.Tensor) -> torch.Tensor:
    """Right-multiply each 64-block of the last dim by T (64x64)."""
    prefix = list(x.shape[:-1])
    c = x.shape[-1]
    y = x.reshape(*prefix, c // _BLOCK, _BLOCK)
    y = torch.matmul(y, T)
    return y.reshape(*prefix, c)


def sym_zero_trace_exp(B: torch.Tensor) -> torch.Tensor:
    """exp(B) for symmetric zero-trace B with eigenvalue clipping."""
    B = (B + B.t()) / 2.0
    B = B - torch.eye(B.shape[0], device=B.device) * (B.diag().mean())
    if float(B.norm()) < 1e-12:
        return torch.matrix_exp(B)
    ev, evec = torch.linalg.eigh(B)
    log2_4 = math.log(2.0) / 4.0
    ev2 = ev.clamp(-log2_4, log2_4)
    B2 = (evec * ev2).mm(evec.t())
    return torch.matrix_exp(B2)


def ste_output_loss(
    weight_smooth: torch.Tensor,
    x_s: torch.Tensor,
    ref: torch.Tensor,
    T: torch.Tensor,
    gram: torch.Tensor,
    offs: tuple,
    et: float,
    am: float,
    mrr: float,
    mrb: int,
) -> torch.Tensor:
    """STE round-trip quantized output error (normalized by std MSE)."""
    T_inv = torch.linalg.inv(T).t()
    W_T = apply_T_block(weight_smooth, T_inv)
    X_T = apply_T_block(x_s, T)
    W_T_d = W_T.detach()
    X_T_d = X_T.detach()
    # weight greedy encode (fast path with offsets) on the transformed coord
    w_params = sol._dense_to_hif4(
        W_T_d,
        importance=None,
        group_gram=None,
        search_offsets=offs,
        error_threshold=et,
        accept_margin=am,
        max_refine_ratio=0.0,
        max_refine_blocks=0,
    )
    w_hat = sol._dequantize_hif4(w_params).to(torch.float32)
    a_params = sol._dense_to_hif4(
        X_T_d,
        importance=None,
        group_gram=None,
        search_offsets=offs,
        error_threshold=et,
        accept_margin=am,
        max_refine_ratio=0.0,
        max_refine_blocks=0,
    )
    a_hat = sol._dequantize_hif4(a_params).to(torch.float32)
    # STE: quantized output + straight-through gradient wrt the continuous input.
    w_hat_ste = w_hat + (W_T - W_T_d)
    a_hat_ste = a_hat + (X_T - X_T_d)
    out = a_hat_ste @ w_hat_ste.t()
    loss = (out - ref).square().mean()
    return loss


def probe_direction(in_features: int, out_features: int, label: str) -> None:
    weight_smooth, x_s, ref, std_mse = build_clean_state(in_features, out_features)
    # gram from clean transformed activations
    gram = (x_s.t() @ x_s) / max(float(x_s.shape[0]), 1e-9)
    offs = _WEIGHT_OFFSETS if hasattr(sol, "_WEIGHT_OFFSETS") else (-1, 1, 2, 3)
    offs = tuple(int(o) for o in offs)
    et = 0.0
    am = 0.0
    mrr = 0.0
    mrb = 0

    # baseline (T = I) loss, greedy encode with offsets
    T0 = torch.eye(_BLOCK, device=device)
    loss0 = ste_output_loss(
        weight_smooth, x_s, ref, T0, gram, offs, et, am, mrr, mrb
    )
    print(
        f"[{label}] baseline(T=I) norm output loss = {float(loss0 / std_mse):.6f}"
    )

    # train T = T1 x T2
    B1 = torch.zeros(8, 8, device=device, requires_grad=True)
    B2 = torch.zeros(8, 8, device=device, requires_grad=True)
    opt = torch.optim.Adam([B1, B2], lr=0.01)
    reg = 1e-3
    steps = 32
    t0 = time.perf_counter()
    for step in range(steps):
        T1 = sym_zero_trace_exp(B1)
        T2 = sym_zero_trace_exp(B2)
        T = torch.kron(T1, T2)
        loss = ste_output_loss(
            weight_smooth, x_s, ref, T, gram, offs, et, am, mrr, mrb
        )
        reg_loss = reg * (B1.square().mean() + B2.square().mean())
        total = loss / std_mse + reg_loss
        opt.zero_grad()
        total.backward()
        g = torch.cat([B1.grad.flatten(), B2.grad.flatten()])
        norm = g.norm().clamp_min(1e-12)
        B1.grad.mul_(1.0 / norm)
        B2.grad.mul_(1.0 / norm)
        opt.step()
        if step in (0, 7, 15, 31):
            print(
                f"    step {step + 1}: norm output loss = {float(loss.detach() / std_mse):.6f}"
            )
    dt = time.perf_counter() - t0
    print(f"[{label}] 32-step STE training wall: {dt:.1f}s ({dt / steps:.3f}s/step)")


_WEIGHT_OFFSETS = (-1, 1, 2, 3)
probe_direction(768, 768, "narrow-qkv")
probe_direction(4864, 768, "wide-proj-in")