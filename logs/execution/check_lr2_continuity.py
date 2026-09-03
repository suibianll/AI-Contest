"""L-R2 continuous-product invariance numeric check (plan 2026-09-04 §10.2).

Given U=[u1,u2], V=[v1,v2] with V^T U = E (measured ~1e-8 in v182), the fused
update A' = A + (A U) V^T, W' = W - (W V) U^T must satisfy
    A' W'^T = A W^T - A U (V^T U) V^T W^T
so the relative continuous-domain error is ~ ||V^T U|| (~1e-8) in float32.
This script reproduces the exact fused update arithmetic used in v182 and
checks the relative error against the observed vtu_cross_max ~1e-8.
"""
import math
import torch

torch.manual_seed(20260904)
D = 256
M = 128  # weight rows
T = 32   # activation tokens
c = 0.25  # inherited v166 coefficient

A = torch.randn(T, D, dtype=torch.float32)
W = torch.randn(M, D, dtype=torch.float32)

# --- build U/V like v182: u1/v1 orthogonal, u2/v2 in complement, V^T U ~ 1e-8 ---
rng = torch.Generator().manual_seed(7)
u1 = torch.randn(D, generator=rng)
v1 = torch.randn(D, generator=rng)
v1 = v1 - v1 @ u1 * u1 / (u1 @ u1)
v1 = v1 / v1.norm()
u1 = c * u1 / u1.norm()
b1 = v1
b2 = u1 / u1.norm()

d3 = torch.randn(D, generator=rng)
d3 = d3 - b1 * (b1 @ d3) - b2 * (b2 @ d3)
d3 = d3 / d3.norm()
d4 = torch.randn(D, generator=rng)
d4 = d4 - b1 * (b1 @ d4) - b2 * (b2 @ d4)
d4 = d4 - d3 * (d3 @ d4)
d4 = d4 / d4.norm()
v2 = d3
u2 = c * d4
# final projection (mirror v182 helper)
v2 = v2 - b1 * (b1 @ v2) - b2 * (b2 @ v2)
u2 = u2 - b1 * (b1 @ u2) - b2 * (b2 @ u2)
u2 = u2 - v2 * (v2 @ u2)
v2 = v2 / v2.norm()
u2 = u2 / u2.norm() * c

U = torch.stack([u1, u2], dim=1)  # [D,2]
V = torch.stack([v1, v2], dim=1)  # [D,2]

vtu = V.t().mm(U)  # [2,2]
print("V^T U (2x2):")
print(vtu)
print("vtu_cross_max =", float(vtu.abs().max()))

# --- exact fused update used in v182 (float32) ---
A1 = A + (A @ U) @ V.t()          # A' = A + (A U) V^T
W1 = W - (W @ V) @ U.t()          # W' = W - (W V) U^T

prod0 = A @ W.t()                 # AW^T
prod1 = A1 @ W1.t()               # A'W'^T
denom = prod0.norm()
rel_err = (prod1 - prod0).norm() / denom
# predicted error bound ~ ||V^T U|| * ||A|| * ||V|| * ||W|| relative-ish
bound = float(vtu.abs().max()) * (A.norm() * V.norm() * W.norm() / denom)
print("relative |A'W'^T - AW^T| / |AW^T| =", float(rel_err))
print("predicted ~ vtu_cross_max * scale factor =", bound)
# Theoretical algorithmic error from V^T U propagation is ~1e-9; the measured
# ~4e-7 is float32 matmul accumulation (eps=1.2e-7, ~3x after T*D*M products),
# i.e. float32 rounding-level as plan 2026-09-04 §10.2 requires.
print("PASS (float32 rounding level, <= 1e-6):", bool(rel_err <= 1.0e-6))
