"""Math, joint-objective gradient, state-contract, and isolated-import checks."""

from pathlib import Path
import importlib.util
import os
import subprocess
import sys

import torch


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "evaluator"))
import reference_hif4 as ref

torch.set_num_threads(1)


def load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


m = load(HERE / "candidate" / "solution.py", "a23_math_candidate")

torch.manual_seed(21071)
dim = 8
raw = torch.randn(2, dim, dim)
s = (raw + raw.transpose(-1, -2)) * 0.05
s = s - torch.diag_embed(
    s.diagonal(dim1=-2, dim2=-1).mean(-1, keepdim=True).expand(2, dim)
)
plus, plus_cache = m._a23_exp(s)
minus, _ = m._a23_exp(s, -1.0)
torch.testing.assert_close(
    plus @ minus, torch.eye(dim).expand(2, dim, dim), atol=3e-5, rtol=3e-5
)
assert bool(torch.isfinite(plus).all() and torch.isfinite(minus).all())

projected = m._a23_project(raw)
eigenvalues = torch.linalg.eigvalsh(projected)
assert float(eigenvalues.abs().max()) <= m._A23_SPECTRAL_BOUND + 2e-5
assert float(projected.diagonal(dim1=-2, dim2=-1).sum(-1).abs().max()) <= 2e-5
torch.testing.assert_close(
    projected, projected.transpose(-1, -2), atol=2e-5, rtol=2e-5
)

base = torch.tensor(
    [[-0.14, 0.02, 0.01, 0.00], [0.02, -0.03, 0.04, 0.01],
     [0.01, 0.04, 0.08, -0.02], [0.00, 0.01, -0.02, 0.16]],
    dtype=torch.float32, requires_grad=True,
)
weight = torch.tensor(
    [[0.3, -0.1, 0.2, 0.4], [-0.1, 0.5, 0.2, -0.2],
     [0.2, 0.2, -0.4, 0.1], [0.4, -0.2, 0.1, 0.2]], dtype=torch.float32,
)
weight = (weight + weight.transpose(-1, -2)) * 0.5
value, cache = m._a23_exp(base)
exact = torch.autograd.grad((value * weight).sum(), base, retain_graph=True)[0]
found = m._a23_exp_backward(weight, cache)
torch.testing.assert_close(found, exact, atol=2e-4, rtol=2e-4)

# Validate the hand-derived amax-product gradient against autograd.
q = torch.randn(3, 4 * 64, requires_grad=True)
k = torch.randn(3, 2 * 64, requires_grad=True)
parent_q = m._a23_group_block_amax2(q.detach(), 4, 2) + 0.25
parent_k = m._a23_group_block_amax2(k.detach(), 2, 2) + 0.25
loss, grad_q, grad_k, _, _ = m._a23_qk_product_loss_grad(
    q, k, parent_q, parent_k, 4, 2
)
autograd_q = torch.autograd.grad(loss, (q, k), retain_graph=True)
torch.testing.assert_close(grad_q, autograd_q[0], atol=2e-5, rtol=2e-5)
torch.testing.assert_close(grad_k, autograd_q[1], atol=2e-5, rtol=2e-5)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
qh, kvh, hd, seq = 4, 2, 64, 5
q_quant = torch.randn(seq, qh * hd, device=device)
k_quant = torch.randn(seq, kvh * hd, device=device)
v_quant = torch.randn(seq, kvh * hd, device=device)
q_scale = torch.ones(seq, qh * hd // 16, device=device)
k_scale = torch.ones(seq, kvh * hd // 16, device=device)
v_scale = torch.ones(seq, kvh * hd // 16, device=device)
common = {
    "multiplier": None, "permutation": None, "importance": None,
    "offsets": torch.tensor((-1, 1, 2, 3, 4), dtype=torch.int8),
    "error_threshold": 1e-7, "accept_margin": 0.0,
    "max_refine_ratio": 0.0, "max_refine_blocks": 0, "version": 2,
}
q_state = dict(common, num_heads=qh, head_dim=hd)
k_state = dict(common, center_mode=0, num_heads=kvh, head_dim=hd)
v_state = dict(common, num_heads=kvh, head_dim=hd)
q_state["learned_rotation"] = torch.eye(hd).expand(kvh, hd, hd).clone()
k_state["learned_rotation"] = torch.eye(hd).expand(kvh, hd, hd).clone()
k_state["learned_center"] = torch.zeros(kvh, hd)
q_params = m.hif4_dynamic_quantize_q(q_quant, q_scale, qh, hd, q_state)
k_params = m.hif4_dynamic_quantize_k(k_quant, k_scale, kvh, hd, k_state)
v_params = m.hif4_dynamic_quantize_v(v_quant, v_scale, kvh, hd, v_state)
for state in (q_state, k_state, v_state):
    ref.validate_state(state)
for params, tensor in ((q_params, q_quant), (k_params, k_quant), (v_params, v_quant)):
    ref.validate_hif4_params(params, tensor.shape)
    assert bool(torch.isfinite(m._dequantize_hif4(params)).all())

code = """from pathlib import Path
import sys
namespace = {'__name__': 'standalone_candidate', '__file__': 'solution.py'}
exec(compile(Path(sys.argv[1]).read_text(encoding='utf-8'), 'solution.py', 'exec'), namespace)
apis = ['hif4_calibration_and_quantize_weight', 'hif4_dynamic_quantize_activation',
        'hif4_calibration_attention', 'hif4_dynamic_quantize_q',
        'hif4_dynamic_quantize_k', 'hif4_dynamic_quantize_v']
assert all(callable(namespace[name]) for name in apis)
root = Path(sys.argv[1]).resolve().parents[3]
assert str(root) not in sys.path
assert not any(str(root / 'workbench') in path for path in sys.path)
print('PASS: six APIs imported with isolated Python and external cwd')
"""
completed = subprocess.run(
    [sys.executable, "-I", "-c", code, str(HERE / "candidate" / "solution.py")],
    cwd=os.environ.get("TEMP", str(HERE)), check=True, capture_output=True, text=True,
)
print({
    "exp_and_projection": "PASS", "exp_backward": "PASS",
    "joint_product_gradient": "PASS", "public_contract": "PASS",
    "isolated_import": completed.stdout.strip(),
})
