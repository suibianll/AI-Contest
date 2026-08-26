"""Dev unit check: attention rotation invariance (MHA + GQA)."""

import importlib.util

import torch

spec = importlib.util.spec_from_file_location("sol", "solution.py")
sol = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sol)

torch.manual_seed(0)

# MHA: q [5, 12*64], k [5, 12*64], shared signs -> dot product invariant
q = torch.randn(5, 12 * 64)
k = torch.randn(5, 12 * 64)
signs = sol._attention_rotation_signs(12, 64, 0)
qr = sol._apply_attention_rotation(q, 12, 64, signs)
kr = sol._apply_attention_rotation(k, 12, 64, signs)
d1 = q @ k.T
d2 = qr @ kr.T
print("MHA dot invariance max err:", (d1 - d2).abs().max().item())

# GQA: 12 q heads, 6 kv heads, group_size=2. Q head h uses signs[h//2];
# K head g uses signs[g]. Full q@k with repeat_interleave layout.
q6 = torch.randn(5, 12 * 64)
k6 = torch.randn(5, 6 * 64)
s6 = sol._attention_rotation_signs(6, 64, 1)
qr6 = sol._apply_attention_rotation(q6, 12, 64, s6)
kr6 = sol._apply_attention_rotation(k6, 6, 64, s6)
# reference: expand K to per-q-head layout (head h -> kv head h//2)
k_exp = k6.view(5, 6, 1, 64).expand(5, 6, 2, 64).reshape(5, 12, 64)
kr_exp = kr6.view(5, 6, 1, 64).expand(5, 6, 2, 64).reshape(5, 12, 64)
ref = (q6.view(5, 12, 64) * k_exp).sum(-1)
rot = (qr6.view(5, 12, 64) * kr_exp).sum(-1)
print("GQA dot invariance max err:", (ref - rot).abs().max().item())
print(
    "signs shape:",
    tuple(signs.shape),
    "unique values:",
    sorted(set(signs.flatten().tolist())),
)
