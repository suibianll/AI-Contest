"""End-to-end objective gradient and isolated six-API import checks."""
from pathlib import Path
import importlib.util
import json
import os
import subprocess
import sys
import torch

HERE = Path(__file__).resolve().parent
torch.set_num_threads(1)
spec = importlib.util.spec_from_file_location("a22_math", HERE / "solution.py")
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
torch.manual_seed(21071)
errors = []
for dim in [32, 64]:
    for zero in [False, True]:
        raw = torch.zeros(2, dim, dim) if zero else torch.randn(2, dim, dim) * 0.005
        raw.requires_grad_()
        s = (raw + raw.transpose(-1, -2)) * 0.5
        q, k = torch.randn(5, 4, dim), torch.randn(9, 2, dim)
        denominator_q = q.reshape(5, -1, 64).abs().amax(-1, keepdim=True)
        denominator_k = k.reshape(9, -1, 64).abs().amax(-1, keepdim=True)
        ep, em = torch.matrix_exp(s), torch.matrix_exp(-s)
        yq = torch.stack([q[:, h] @ ep[h // 2] for h in range(4)], 1).flatten(1)
        yk = torch.stack([k[:, h] @ em[h] for h in range(2)], 1).flatten(1)
        loss = (yq.reshape(5, -1, 64).abs().amax(-1, keepdim=True) / denominator_q).square().mean()
        loss = loss + (yk.reshape(9, -1, 64).abs().amax(-1, keepdim=True) / denominator_k).square().mean()
        exact = torch.autograd.grad(loss, raw)[0]
        sp = s.detach()
        ep, cp = m._a21_exp(sp)
        em, cm = m._a21_exp(sp, -1)
        xq, xk = q.flatten(1), k.flatten(1)
        _, gq = m._a21_scale_loss_grad(m._a2_apply_group_rotation(xq, 4, ep), denominator_q)
        _, gk = m._a21_scale_loss_grad(m._a2_apply_group_rotation(xk, 2, em), denominator_k)
        manual = m._a21_exp_backward(m._a21_matrix_grad(xq, gq, 4, 2), cp)
        manual += m._a21_exp_backward(m._a21_matrix_grad(xk, gk, 2, 2), cm)
        torch.testing.assert_close(manual, exact, atol=2e-5, rtol=1e-4)
        errors.append(float((manual - exact).abs().max()))

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
result = {"full_objective_gradient_errors": errors, "isolated_import": completed.stdout.strip()}
(HERE / "math-and-import.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
print(json.dumps(result, indent=2))
