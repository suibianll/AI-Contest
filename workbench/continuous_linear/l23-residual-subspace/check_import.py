"""Detached single-file import check for the L23 candidate.

Imports the six official APIs from the isolated solution.py copy (no repo
paths on sys.path) and validates weight/activation legality; attention is
frozen (L4 parent side) so it is only import-checked here.
"""

from __future__ import annotations

import hashlib
import importlib.util
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))

sys.path.insert(0, os.path.join(ROOT, "evaluator"))
import reference_hif4 as ref  # noqa: E402

SOLUTION = os.path.join(HERE, "candidate", "solution.py")
spec = importlib.util.spec_from_file_location("detached_solution", SOLUTION)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

apis = [
    "hif4_calibration_and_quantize_weight",
    "hif4_dynamic_quantize_activation",
    "hif4_calibration_attention",
    "hif4_dynamic_quantize_q",
    "hif4_dynamic_quantize_k",
    "hif4_dynamic_quantize_v",
]
for name in apis:
    assert callable(getattr(module, name, None)), f"missing API {name}"
print("six APIs importable:", apis)

sha = hashlib.sha256(open(SOLUTION, "rb").read()).hexdigest().upper()
print("sha256:", sha)

# legal weight calibration smoke mimicking official contract shapes
torch.manual_seed(0)
tokens, channels = 64, 256
fp4_grid = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
blocks = channels // 16


def make_pair(n):
    dense = torch.randn(n, channels)
    grouped = dense.unflatten(-1, (blocks, 16))
    scale = (grouped.abs().amax(-1) / 6.0).clamp(min=1e-6, max=1e6)
    normalized = grouped / scale[..., None]
    sign = normalized.sign()
    index = torch.argmin((normalized.abs().unsqueeze(-1) - fp4_grid).abs(), dim=-1)
    quant = (sign * fp4_grid[index]).flatten(-2, -1)
    return quant, scale


# official v189-style contract: weight + paired activation list
w_quant, w_scale = make_pair(16)
act_pairs = [make_pair(40) for _ in range(2)]
state = module.hif4_calibration_and_quantize_weight(w_quant, w_scale, act_pairs)
ref.validate_hif4_params(state["weight_params"], (16, channels))
print("weight calibration params legal (validation PASS)")

# activation dynamic smoke on the paired calibration activation
act_quant, act_scale = act_pairs[0]
act_state = module.hif4_dynamic_quantize_activation(act_quant, act_scale, state["activation_state"])
ref.validate_hif4_params(act_state, act_quant.shape)
print("activation params legal")

print("DETACHED IMPORT CHECK PASS")