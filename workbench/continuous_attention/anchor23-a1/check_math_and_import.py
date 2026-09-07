"""A23 CPU checks: product-objective gradient vs autograd + isolated import."""
from pathlib import Path
import importlib.util
import json
import os
import subprocess
import sys
import torch

HERE = Path(__file__).resolve().parent
torch.set_num_threads(1)
spec = importlib.util.spec_from_file_location("a23_math", HERE / "solution.py")
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
torch.manual_seed(21071)
errors = []
# Head-aligned product gradient vs autograd (small geometries, both roles).
for qh, kh, dim, tq, tk in [(4, 4, 256, 5, 5), (4, 2, 128, 6, 9), (2, 2, 64, 7, 5)]:
    raw_q = torch.randn(tq, qh * dim) * 0.4
    raw_k = torch.randn(tk, kh * dim) * 0.4
    xq = raw_q.clone().requires_grad_()
    xk = raw_k.clone().requires_grad_()
    nb = dim // 64
    hpg = qh // kh
    aq = xq.reshape(tq, qh, nb, 64).abs().amax(-1).square().reshape(tq, kh, hpg, nb).mean(dim=(0, 2))
    ak = xk.reshape(tk, kh, nb, 64).abs().amax(-1).square().mean(0)
    dq = (aq.detach() * 1.13 + 0.07).clamp_min(1e-12)
    dk = (ak.detach() * 0.91 + 0.07).clamp_min(1e-12)
    loss = (aq * ak / (dq * dk)).mean()
    gq, gk = torch.autograd.grad(loss, (xq, xk))
    _, man_q, man_k = m._a23_product_loss_grad(xq.detach(), xk.detach(), qh, kh, dim, dq, dk)
    torch.testing.assert_close(man_q, gq, atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(man_k, gk, atol=1e-6, rtol=1e-5)
    errors.append(float((man_q - gq).abs().max()))
# Constant Q*c / K/c product invariance (the property the additive target lacks).
xq = torch.randn(6, 2 * 64) * 0.4
xk = torch.randn(6, 2 * 64) * 0.4
scale = torch.full((64,), 1.31)
xq_s = xq.clone()
xq_s[:, :64] = xq[:, :64] * scale
xk_s = xk.clone()
xk_s[:, :64] = xk[:, :64] / scale
a0q, a0k = m._a23_moments(xq, xk, 2, 2, 64)
a1q, a1k = m._a23_moments(xq_s, xk_s, 2, 2, 64)
assert abs(float((a1q * a1k / (a0q * a0k)).mean()) - 1.0) < 1e-5

code = """from pathlib import Path
import sys
namespace = {'__name__': 'standalone_candidate', '__file__': 'solution.py'}
exec(compile(Path(sys.argv[1]).read_text(encoding='utf-8'), 'solution.py', 'exec'), namespace)
apis = ['hif4_calibration_and_quantize_weight', 'hif4_dynamic_quantize_activation', 'hif4_calibration_attention', 'hif4_dynamic_quantize_q', 'hif4_dynamic_quantize_k', 'hif4_dynamic_quantize_v']
assert all(callable(namespace[name]) for name in apis)
root = Path(sys.argv[1]).resolve().parents[3]
assert str(root) not in sys.path
assert not any(str(root / 'workbench') in path for path in sys.path)
print('PASS: six APIs imported with isolated Python and external cwd')
"""
completed = subprocess.run([sys.executable, "-I", "-c", code, str(HERE / "solution.py")],
                           cwd=os.environ["TEMP"], check=True, capture_output=True, text=True)
result = {"product_gradient_vs_autograd_max_errors": errors,
          "product_invariance": "PASS: Q*c with K/c leaves the joint product unchanged",
          "isolated_import": completed.stdout.strip()}
(HERE / "math-and-import.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
print(json.dumps(result, indent=2))
