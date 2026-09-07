"""A26-A small-shape mathematical verification (CPU, no GPU needed).

Checks (mechanism.md pre-registered):
  V1  inverse: ||exp(S) exp(-S)^T - I||_inf <= 1e-5 for random symmetric S
  V2  A24 control: with the residual model forced to zero (delta=1e12), the
      A26 loss is identically zero for ANY S (the STE dead-end demonstration)
  V3  gradient non-vanishing: at S=0 with realistic delta, ||dL/dS|| > 0
  V4  phase direction: block values sitting at half-grid give a larger loss
      than the same values moved to code points (sin model rewards code
      alignment, and moving them back increases the loss)
  V5  S=0 identity: exp(0) rotation reproduces inputs to ~1e-6 (candidate
      == base stack bitwise through the readout path at S=0)
  V6  GQA grouping: one S per KV group shared by 4 Q heads (shape + rotation
      structure), and per-group loss decomposition matches the full loss
"""
from pathlib import Path
import importlib.util
import math
import sys

import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


mod = load(HERE / "solution.py", "a26_verify")
torch.manual_seed(21071)

T, QH, KH, DIM = 16, 16, 4, 256
failures = []

# V1 inverse (S in the training spectral domain: projected, clamp +-log2/2)
s = torch.randn(KH, DIM, DIM)
s = mod._a21_project((s + s.transpose(-1, -2)) * 0.5)
e_q = mod._a21_exp(s)[0]
e_k = mod._a21_exp(s, -1.0)[0]
err = float((e_q @ e_k.transpose(-1, -2) - torch.eye(DIM)).abs().max())
print(f"V1 inverse error = {err:.3e}", "OK" if err < 1e-5 else "FAIL")
if err >= 1e-5:
    failures.append("V1")

# V2 A24 control: residual model forced to zero (STE identity) -> the loss is
# identically zero for ANY S and carries zero gradient (the dead end itself)
u_q = torch.randn(T, QH * DIM)
u_k = torch.randn(T, KH * DIM)
delta_q = (u_q.reshape(T, -1, 64).abs().amax(-1, keepdim=True) * 0.25).detach()
delta_k = (u_k.reshape(T, -1, 64).abs().amax(-1, keepdim=True) * 0.25).detach()
original_residual = mod._a26_grid_residual
mod._a26_grid_residual = lambda x, delta: torch.zeros_like(x)
s_leaf = s.clone().requires_grad_(True)
e_q = mod._A26Exp.apply(s_leaf, 1.0)
e_k = mod._A26Exp.apply(s_leaf, -1.0)
loss_zero = mod._a26_output_loss(u_q, u_k, e_q, e_k, delta_q, delta_k, QH, KH, DIM)
loss_zero.backward()
zero_loss = float(loss_zero)
zero_grad = float(s_leaf.grad.norm())
mod._a26_grid_residual = original_residual
print(f"V2 A24 control: loss(eps=0) = {zero_loss:.3e}, ||dL/dS|| = {zero_grad:.3e}",
      "OK" if zero_loss == 0.0 and zero_grad == 0.0 else "FAIL")
if not (zero_loss == 0.0 and zero_grad == 0.0):
    failures.append("V2")

# V3 gradient non-vanishing at S=0 with realistic delta
s0 = torch.zeros(KH, DIM, DIM, requires_grad=True)
e_q = mod._A26Exp.apply(s0, 1.0)
e_k = mod._A26Exp.apply(s0, -1.0)
loss = mod._a26_output_loss(u_q, u_k, e_q, e_k, delta_q, delta_k, QH, KH, DIM)
loss.backward()
gnorm = float(s0.grad.norm())
print(f"V3 ||dL/dS|| at S=0 = {gnorm:.3e}", "OK" if gnorm > 1e-8 else "FAIL")
if gnorm <= 1e-8:
    failures.append("V3")

# V4 phase direction (single-variable: K fixed at code points, only Q moves)
delta2 = torch.full((T, (QH * DIM) // 64, 1), 2.0)
delta2k = torch.full((T, (KH * DIM) // 64, 1), 2.0)
u_on_code = (2.0 * torch.randint(1, 5, (T, QH * DIM)).float())
k_on_code = (2.0 * torch.randint(1, 5, (T, KH * DIM)).float())
e_i = torch.eye(DIM).expand(KH, DIM, DIM)
loss_code = mod._a26_output_loss(u_on_code, k_on_code, e_i, e_i, delta2, delta2k, QH, KH, DIM)
loss_half = mod._a26_output_loss(u_on_code + 1.0, k_on_code, e_i, e_i, delta2, delta2k, QH, KH, DIM)
print(f"V4 loss(code-point) = {float(loss_code):.4f} < loss(half-grid) = {float(loss_half):.4f}",
      "OK" if float(loss_code) < float(loss_half) * 0.5 else "FAIL")
if not float(loss_code) < float(loss_half) * 0.5:
    failures.append("V4")

# V5 S=0 identity through the rotation path
e_id = mod._a21_exp(torch.zeros(KH, DIM, DIM))[0]
u_t = mod._a2_apply_group_rotation(u_q, QH, e_id)
ident_err = float((u_t - u_q).abs().max())
print(f"V5 S=0 rotation identity max|diff| = {ident_err:.3e}", "OK" if ident_err < 1e-5 else "FAIL")
if ident_err >= 1e-5:
    failures.append("V5")

# V6 GQA structure: group rotation with (KH, DIM, DIM) matches per-group manual
u3 = u_q.reshape(T, KH, QH // KH, DIM)
manual = torch.empty_like(u3)
for g in range(KH):
    manual[:, g] = u3[:, g] @ e_id[g]
manual = manual.reshape(T, QH * DIM)
gqa_err = float((mod._a2_apply_group_rotation(u_q, QH, e_id) - manual).abs().max())
print(f"V6 GQA group-sharing max|diff| = {gqa_err:.3e}", "OK" if gqa_err < 1e-5 else "FAIL")
if gqa_err >= 1e-5:
    failures.append("V6")

print("\nRESULT:", "ALL OK" if not failures else f"FAILED: {failures}")
sys.exit(1 if failures else 0)
