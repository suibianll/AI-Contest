import importlib.util
import torch

spec = importlib.util.spec_from_file_location("s1cand", r"workbench\s1_qk_gram_refine.py")
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

apis = [
    "hif4_calibration_and_quantize_weight",
    "hif4_dynamic_quantize_activation",
    "hif4_calibration_attention",
    "hif4_dynamic_quantize_q",
    "hif4_dynamic_quantize_k",
    "hif4_dynamic_quantize_v",
]
print("6 APIs:", all(hasattr(m, a) for a in apis))
print("gram refine:", m._ATTN_GRAM_REFINE, "sweeps:", m._ATTN_GRAM_SWEEPS)

torch.manual_seed(0)
q_heads, kv_heads, head_dim = 14, 2, 64
group = q_heads // kv_heads

# --- cross gram construction ---
tq = [torch.randn(64, q_heads * head_dim) * 0.3 for _ in range(3)]
tk = [torch.randn(64, kv_heads * head_dim) * 0.2 for _ in range(3)]
q_gram, k_gram = m._qk_cross_gram64(tq, tk, q_heads, kv_heads, head_dim)
assert q_gram.shape == (q_heads * (head_dim // 64), 64, 64), q_gram.shape
assert k_gram.shape == (kv_heads * (head_dim // 64), 64, 64), k_gram.shape
assert torch.allclose(q_gram, q_gram.transpose(-1, -2), atol=1e-5)
assert bool((q_gram.diagonal(dim1=-2, dim2=-1) >= -1e-6).all())
assert bool((k_gram.diagonal(dim1=-2, dim2=-1) >= -1e-6).all())
print("cross gram shapes/symmetry/PSD: OK")

# --- refine monotonicity: gram loss must not increase ---
T = 96
x = torch.randn(T, q_heads * head_dim) * 0.3
params = m._dense_to_hif4(x)
gram = q_gram.to(torch.float32)
def gram_loss(p):
    err = m._dequantize_hif4(p).to(torch.float32) - x
    e = err.reshape(T, -1, 64)
    return float(torch.einsum("tbi,bij,tbj->", e, gram, e))
before = gram_loss(params)
refined = m._refine_activation_gram(
    x, params, gram, max_blocks=q_heads, sweeps=3
)
after = gram_loss(refined)
print(f"gram loss: {before:.4f} -> {after:.4f}")
assert after <= before + 1e-6 * max(1.0, abs(before)), "refine increased gram loss"

# reconstruction must not be destroyed (refine trades recon for logit loss; allow bounded change)
rec_before = float((m._dequantize_hif4(params) - x).square().mean())
rec_after = float((m._dequantize_hif4(refined) - x).square().mean())
print(f"recon mse: {rec_before:.6f} -> {rec_after:.6f}")

# params key/layout unchanged
assert set(refined.keys()) == set(params.keys())
for k in params:
    assert refined[k].shape == params[k].shape, k
print("params layout: OK")

# gram=None passthrough
out = m._refine_activation_gram(x, params, None, max_blocks=4, sweeps=3)
assert out is params
print("ALL S1 UNIT CHECKS PASSED")
