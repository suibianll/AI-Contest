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


m = load(HERE / "candidate" / "solution.py", "tri_math_candidate")

# The two rank-1 blocks are disjoint, so Q*T and K*T^{-T} preserve every GQA
# dot product and N^2 is exactly zero in the full 4x4 block matrix.
torch.manual_seed(21071)
groups, q_heads, head_dim, tokens = 2, 4, 256, 5
left = torch.randn(groups, 2, 64)
right = torch.randn(groups, 2, 64)
directions = torch.einsum("gpi,gpj->gpij", left, right)
directions = directions / directions.flatten(2).norm(dim=-1, keepdim=True).unsqueeze(-1)
blocks = directions * 0.03
q = torch.randn(tokens, q_heads * head_dim)
k = torch.randn(tokens, groups * head_dim)
q_new = m._apply_attn_triangular_transform(q, q_heads, blocks, False)
k_new = m._apply_attn_triangular_transform(k, groups, blocks, True)
q_heads_view = q.reshape(tokens, q_heads, head_dim)
k_heads_view = k.reshape(tokens, groups, head_dim).repeat_interleave(q_heads // groups, dim=1)
q_new_view = q_new.reshape(tokens, q_heads, head_dim)
k_new_view = k_new.reshape(tokens, groups, head_dim).repeat_interleave(q_heads // groups, dim=1)
torch.testing.assert_close(
    torch.einsum("thi,shi->ths", q_new_view, k_new_view),
    torch.einsum("thi,shi->ths", q_heads_view, k_heads_view),
    atol=2e-5,
    rtol=2e-5,
)
for group_index in range(groups):
    n_matrix = torch.zeros(256, 256)
    n_matrix[:64, 64:128] = blocks[group_index, 0]
    n_matrix[128:192, 192:256] = blocks[group_index, 1]
    torch.testing.assert_close(n_matrix @ n_matrix, torch.zeros_like(n_matrix), atol=2e-5, rtol=2e-5)
    t_matrix = torch.eye(256) + n_matrix
    inverse_matrix = torch.eye(256) - n_matrix
    torch.testing.assert_close(t_matrix @ inverse_matrix, torch.eye(256), atol=2e-5, rtol=2e-5)

# Nonzero blocks must be rank one; the disjoint support makes N^2 exactly zero
# in the full 4x4 block matrix.
for group_index in range(groups):
    for pair_index in range(2):
        assert int(torch.linalg.matrix_rank(blocks[group_index, pair_index]).item()) == 1

# Exercise the public Q/K/V contract with an actual nonzero compiled transport.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
seq = 4
q_quant = torch.randn(seq, q_heads * head_dim, device=device)
k_quant = torch.randn(seq, groups * head_dim, device=device)
v_quant = torch.randn(seq, groups * head_dim, device=device)
q_scale = torch.ones(seq, q_heads * head_dim // 16, device=device)
k_scale = torch.ones(seq, groups * head_dim // 16, device=device)
v_scale = torch.ones(seq, groups * head_dim // 16, device=device)
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
q_state = dict(common, num_heads=q_heads, head_dim=head_dim)
k_state = dict(common, center_mode=0, num_heads=groups, head_dim=head_dim)
q_state["triangular_blocks"] = blocks.clone()
k_state["triangular_blocks"] = blocks.clone()
q_state["triangular_inverse"] = False
k_state["triangular_inverse"] = True
v_state = dict(common, num_heads=groups, head_dim=head_dim)
q_params = m.hif4_dynamic_quantize_q(q_quant, q_scale, q_heads, head_dim, q_state)
k_params = m.hif4_dynamic_quantize_k(k_quant, k_scale, groups, head_dim, k_state)
v_params = m.hif4_dynamic_quantize_v(v_quant, v_scale, groups, head_dim, v_state)
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
    cwd=os.environ.get("TEMP", str(HERE)),
    check=True,
    capture_output=True,
    text=True,
)
print({
    "triangular_inverse_and_dot_invariance": "PASS",
    "rank1_disjoint_blocks": "PASS",
    "public_contract": "PASS",
    "isolated_import": completed.stdout.strip(),
})
