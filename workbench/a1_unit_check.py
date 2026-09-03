import importlib.util
import torch

spec = importlib.util.spec_from_file_location("a1cand", r"workbench\a1_matrix_smooth4.py")
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

torch.manual_seed(0)
q_heads, kv_heads, head_dim = 14, 2, 64
group = q_heads // kv_heads
T = 256
q_samples = [torch.randn(8, q_heads * head_dim) * 0.3 for _ in range(4)]
k_samples = [torch.randn(8, kv_heads * head_dim) * 0.2 for _ in range(4)]

fit = m._fit_attention_pair_matrix_smooth(
    q_samples, k_samples, {}, {}, q_heads, kv_heads, head_dim
)
assert fit is not None, "fit returned None"
q_matrix, k_matrix, q_imp, k_imp = fit
print("q_matrix", tuple(q_matrix.shape), "k_matrix", tuple(k_matrix.shape))
print("q_importance", tuple(q_imp.shape), "k_importance", tuple(k_imp.shape))
assert q_matrix.shape == (q_heads, head_dim // 4, 4, 4)
assert k_matrix.shape == (kv_heads, head_dim // 4, 4, 4)
assert q_imp.shape == (q_heads * head_dim,)
assert k_imp.shape == (kv_heads * head_dim,)

# continuous-domain invariance: q_matrix @ k_matrix^T == I within each GQA group
prod = q_matrix.reshape(kv_heads, group, head_dim // 4, 4, 4) @ k_matrix.unsqueeze(1).transpose(-1, -2)
eye = torch.eye(4).expand_as(prod)
err = (prod - eye).abs().max().item()
print("q@k^T max abs err vs I:", err)
assert err < 1e-4, "pair transform not inverse-transpose"

# deployed path invariance: apply to dense Q/K and compare logits
q_dense = torch.randn(64, q_heads * head_dim)
k_dense = torch.randn(64, kv_heads * head_dim)
q_t = m._apply_attention_pair_transform(q_dense, q_heads, head_dim, q_matrix)
k_t = m._apply_attention_pair_transform(k_dense, kv_heads, head_dim, k_matrix)
q_r = q_t.reshape(64, kv_heads, group, head_dim)
logit_new = torch.einsum("tghd,tgd->thg", q_r, k_t.reshape(64, kv_heads, head_dim))
logit_old = torch.einsum(
    "tghd,tgd->thg",
    q_dense.reshape(64, kv_heads, group, head_dim),
    k_dense.reshape(64, kv_heads, head_dim),
)
diff = (logit_new - logit_old).abs().max().item()
rel = diff / max(logit_old.abs().max().item(), 1e-12)
print("QK logits max abs diff:", diff, "rel:", rel)
assert rel < 1e-4, "continuous QK^T changed"

# 2x2 fallback when head_dim % 4 != 0 (head_dim=6)
fit6 = m._fit_attention_pair_matrix_smooth(
    [torch.randn(8, q_heads * 6) for _ in range(2)],
    [torch.randn(8, kv_heads * 6) for _ in range(2)],
    {}, {}, q_heads, kv_heads, 6,
)
if fit6 is not None:
    assert fit6[0].shape == (q_heads, 3, 2, 2)
    print("head_dim=6 falls back to 2x2: OK")
else:
    print("head_dim=6 rejected (odd-pair rule)")
print("ALL NUMERIC CHECKS PASSED")
