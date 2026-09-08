"""Math, state-contract, finite-output, and isolated-import checks."""

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


m = load(HERE / "candidate" / "solution.py", "diag_math_candidate")

# Closed-form solution: zero mean is imposed before the fixed log(2)/2 bound.
torch.manual_seed(21071)
kv, dim, q_heads = 3, 8, 6
parent_q = {"learned_rotation": None}
parent_k = {"learned_rotation": None, "learned_center": None}
a = torch.rand(kv, dim) * 3.0 + 0.1
b = torch.rand(kv, dim) * 3.0 + 0.1
q_state, k_state, info = m._attn_diag_reciprocal_balance(
    {"q_state": parent_q, "k_state": parent_k, "v_state": {}},
    a, b, q_heads, kv, dim,
)
d = info["d"]
assert d.shape == (kv, dim)
assert bool(torch.isfinite(d).all())
assert float(d.abs().max()) <= 0.5 * torch.log(torch.tensor(2.0)) + 1e-7
expected = 0.25 * torch.log((b + 1e-12) / (a + 1e-12))
expected = expected - expected.mean(dim=1, keepdim=True)
expected = expected.clamp(-0.5 * torch.log(torch.tensor(2.0)), 0.5 * torch.log(torch.tensor(2.0)))
torch.testing.assert_close(d, expected, atol=1e-6, rtol=1e-6)
assert "diag_scale" in q_state and "diag_scale" in k_state

# Exact Jacobian-energy reference on a small non-HiF4 shape.  The helper
# averages causal and non-causal squared Jacobian energies.
tokens, small_dim, small_qh, small_kv = 5, 4, 4, 2
q = torch.randn(tokens, small_qh * small_dim)
k = torch.randn(tokens, small_kv * small_dim)
v = torch.randn(tokens, small_kv * small_dim)
q_energy, k_energy = m._attn_diag_output_sensitivity(
    q, k, v, small_qh, small_kv, small_dim
)

def output_fn(q_value, k_value):
    causal = m._attention_forward(
        q_value, k_value, v, small_qh, small_kv, small_dim, True
    )
    noncausal = m._attention_forward(
        q_value, k_value, v, small_qh, small_kv, small_dim, False
    )
    return torch.cat((causal.reshape(-1), noncausal.reshape(-1)))

jac_q, jac_k = torch.autograd.functional.jacobian(
    output_fn, (q, k), vectorize=True
)
exact_q = jac_q.square().sum(dim=tuple(range(jac_q.ndim - q.ndim))) / 2.0
exact_k = jac_k.square().sum(dim=tuple(range(jac_k.ndim - k.ndim))) / 2.0
torch.testing.assert_close(
    exact_q.reshape(tokens, small_qh, small_dim),
    q_energy.permute(1, 0, 2),
    atol=3e-5,
    rtol=3e-5,
)
torch.testing.assert_close(
    exact_k.reshape(tokens, small_kv, small_dim),
    k_energy.permute(1, 0, 2),
    atol=3e-5,
    rtol=3e-5,
)

# Public dynamic contract, including the new fixed post-transform vector.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
qh, kvh, hd, seq = 4, 2, 64, 9
q_quant = torch.randn(seq, qh * hd, device=device)
k_quant = torch.randn(seq, kvh * hd, device=device)
v_quant = torch.randn(seq, kvh * hd, device=device)
q_scale = torch.ones(seq, qh * hd // 16, device=device)
k_scale = torch.ones(seq, kvh * hd // 16, device=device)
v_scale = torch.ones(seq, kvh * hd // 16, device=device)
common = {
    "multiplier": None,
    "permutation": None,
    "importance": None,
    "offsets": torch.tensor((-1, 1, 2, 3, 4), dtype=torch.int8),
    "error_threshold": 1e-7,
    "accept_margin": 0.0,
    "max_refine_ratio": 0.0,
    "max_refine_blocks": 0,
    "version": 2,
}
q_contract = dict(common, num_heads=qh, head_dim=hd)
k_contract = dict(common, center_mode=0, num_heads=kvh, head_dim=hd)
q_contract["diag_scale"] = torch.ones(qh * hd)
k_contract["diag_scale"] = torch.ones(kvh * hd)
q_params = m.hif4_dynamic_quantize_q(q_quant, q_scale, qh, hd, q_contract)
k_params = m.hif4_dynamic_quantize_k(k_quant, k_scale, kvh, hd, k_contract)
v_contract = dict(common, num_heads=kvh, head_dim=hd)
v_params = m.hif4_dynamic_quantize_v(v_quant, v_scale, kvh, hd, v_contract)
for state in (q_contract, k_contract, v_contract):
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
    cwd=os.environ.get("TEMP", str(HERE)),
    check=True,
    capture_output=True,
    text=True,
)
print({
    "closed_form": "PASS",
    "jacobian_energy": "PASS",
    "public_contract": "PASS",
    "isolated_import": completed.stdout.strip(),
})
