"""Diagnostic: compare the mechanism's predicted cost with the true dJ and dL.

Read-only.  Not a version artifact; used to attribute a verify.py failure.
"""

import importlib.util
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


candidate = load(HERE / "candidate" / "solution.py", "c")
parent = load(ROOT / "solution.py", "p")

g = torch.Generator().manual_seed(907)
wq = torch.randn((64, 2560), generator=g)
ws = torch.ones((64, 160))
acts = [(torch.randn((12, 2560), generator=g), torch.ones((12, 160)))]
res = candidate.hif4_calibration_and_quantize_weight(wq, ws, acts)
state = res["activation_state"]
weight_params = res["weight_params"]

before = parent.hif4_dynamic_quantize_activation(acts[0][0], acts[0][1], dict(state))
live_state = dict(state)
after = candidate.hif4_dynamic_quantize_activation(
    acts[0][0], acts[0][1], live_state
)
print("diag:", {k: v for k, v in live_state.items() if k.startswith("em1_")})

x_before = candidate._dequantize_hif4(before).to(torch.float64)
x_after = candidate._dequantize_hif4(after).to(torch.float64)
w_hat = candidate._dequantize_hif4(weight_params).to(torch.float64)
x_ref = candidate._dequantize_nvfp4_float32(acts[0][0], acts[0][1]).to(torch.float64)
w_ref = candidate._dequantize_nvfp4_float32(wq, ws).to(torch.float64)

gram = w_hat.t().mm(w_hat)
cross = w_ref.t().mm(w_hat)
target = torch.linalg.lstsq(gram, cross.t().mm(x_ref.t())).solution.t()
print(
    "lstsq residual",
    float((target.mm(gram) - x_ref.mm(cross)).norm() / x_ref.mm(cross).norm()),
)


def objective(x):
    delta = x - target
    return float((delta.mm(gram) * delta).sum())


def loss(x):
    out = x.mm(w_hat.t())
    return float((out - x_ref.mm(w_ref.t())).square().sum())


j_before, j_after = objective(x_before), objective(x_after)
l_before, l_after = loss(x_before), loss(x_after)
print(f"J_before={j_before:.6e} J_after={j_after:.6e} dJ={j_after - j_before:+.6e}")
print(f"L_before={l_before:.6e} L_after={l_after:.6e} dL={l_after - l_before:+.6e}")
print(f"L-J before={l_before - j_before:.6e} after={l_after - j_after:.6e}")
print(f"predicted_cost={live_state.get('em1_predicted_cost')}")

step = 0.25
rows = int(x_before.shape[0])
code_before = torch.round(before["mant"].to(torch.float64) / step).reshape(rows, -1)
code_after = torch.round(after["mant"].to(torch.float64) / step).reshape(rows, -1)
local = (
    before["scale_factor"].to(torch.float64)
    * before["scale_lv2"].to(torch.float64)
    * before["scale_lv3"].to(torch.float64)
).reshape(rows, 40, 16).repeat_interleave(4, dim=2).reshape(rows, -1)
sign = before["sign"].to(torch.float64).reshape(rows, -1)
model_delta = sign * (code_after - code_before) * step * local
print("bookkeeping vs actual max diff", float((model_delta - (x_after - x_before)).abs().max()))
print("delta absmax", float((x_after - x_before).abs().max()))

diff = x_after - x_before
dJ_real = 2.0 * float(((x_before - target).mm(gram) * diff).sum()) + float(
    (diff.mm(gram) * diff).sum()
)
print(f"dJ_realized={dJ_real:+.6e} dJ={j_after - j_before:+.6e}")

metric, h_matrix = candidate._em1_metric(live_state, 2560, torch.device("cpu"))
metric = metric.to(torch.float64)
h_matrix = h_matrix.to(torch.float64)
g_model = (x_before - x_ref).mm(metric) + x_ref.mm(h_matrix)
g_true = (x_before - target).mm(gram)
print(
    "gradient: model norm=%.6e true norm=%.6e rel=%.3e"
    % (
        float(g_model.norm()),
        float(g_true.norm()),
        float((g_model - g_true).norm() / g_true.norm()),
    )
)
print("metric vs gram rel", float((metric - gram).norm() / gram.norm()))
print("h vs (gram - cross) rel", float((h_matrix - (gram - cross)).norm() / (gram - cross).norm()))
cost_realized = 2.0 * float((g_model * diff).sum()) + float((diff.mm(metric) * diff).sum())
print(f"cost_realized_from_model={cost_realized:+.6e}")
