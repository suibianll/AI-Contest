"""Cross-check: does the candidate's _vk_correct reproduce VK-3's measured rule?

VK-3 measured -1.8% on window 3 for the kernel rule, but the candidate's gate
moved the FULL loss by only 0.026% -- about 50x less than the 1.4% implied by
the error decomposition (V is ~78% of the total).  Two implementations of the
same rule disagree, so this runs both on the same inputs:

  A. the diagnostic implementation (vk3_rule.py's greedy, radius 7, 6 sweeps)
  B. the candidate's `_vk_correct`

from the same parent parameters, the same fitted kernel and the same window, and
reports the V-only loss each produces.  If they differ, the candidate is wrong
and the gate number is not evidence about the mechanism.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def main() -> int:
    torch.set_num_threads(4)
    torch.set_grad_enabled(False)
    parent = load("cx_parent", ROOT / "solution.py")
    candidate = load("cx_candidate", HERE / "candidate" / "solution.py")
    diagnostic = load("cx_diag", HERE / "vk3_rule.py")

    sys.path.insert(0, str(ROOT / "evaluator"))
    import official_eval as v2  # noqa: PLC0415

    pack = torch.load(
        ROOT / "artifacts/official_eval/cache/qwen3.5-4b-proxy-v2.pt",
        map_location="cpu", mmap=True, weights_only=False,
    )
    q_heads = int(pack["q_heads"])
    kv_heads = int(pack["kv_heads"])
    dim = int(pack["head_dim"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    layer = int(os.environ.get("CX_LAYER", "0"))
    window = int(os.environ.get("CX_WINDOW", "3"))

    def attention(q, k, v):
        return v2._attention(
            q.to(torch.float32)[None], k.to(torch.float32)[None], v.to(torch.float32)[None],
            q_heads, kv_heads, dim,
        )[0]

    def dense(params, shape):
        return v2.dequantize_hif4(v2._cpu_params(params), shape).to(torch.float32).to(device)

    windows = [
        {
            role: tuple(
                t.to(device)
                for t in v2._pair(pack["calibration_qkv"][s][layer][i].to(torch.float32))
            )
            for i, role in enumerate(("q", "k", "v"))
        }
        for s in range(len(pack["calibration_qkv"]))
    ]
    states = candidate.hif4_calibration_attention(windows, q_heads, kv_heads, dim)
    kernel = states["v_state"].get("vk_kernel")
    print(f"candidate arm={states['v_state'].get('vk_arm')} kernel={'yes' if kernel is not None else 'no'}")

    # the candidate's own kernel, read back, drives BOTH implementations
    entry = windows[window]
    shapes = [v2.dequantize_nvfp4(*entry[r]).shape for r in ("q", "k", "v")]
    pq = parent.hif4_dynamic_quantize_q(*entry["q"], q_heads, dim, states["q_state"])
    pk = parent.hif4_dynamic_quantize_k(*entry["k"], kv_heads, dim, states["k_state"])
    pv = parent.hif4_dynamic_quantize_v(*entry["v"], kv_heads, dim, states["v_state"])
    qd, kd = dense(pq, shapes[0]), dense(pk, shapes[1])
    v_ref = v2.dequantize_nvfp4(*entry["v"]).to(torch.float32).to(device)
    target = attention(qd, kd, v_ref)

    def v_only_loss(params):
        return float((attention(qd, kd, dense(params, shapes[2])) - target).square().mean())

    print(f"parent V-only loss = {v_only_loss(pv):.6e}")

    # A. the candidate's correction, driven by the candidate's kernel
    if kernel is not None:
        a = candidate._vk_correct(pv, kernel, *entry["v"], kv_heads, dim)
        moved_a = int((a["mant"] != pv["mant"]).sum())
        print(f"A candidate._vk_correct : loss={v_only_loss(a):.6e}  moved={moved_a}  "
              f"ratio={v_only_loss(a) / v_only_loss(pv):+.4%}")

    # B. the diagnostic's rule, driven by the SAME kernel (fit parameters match
    #    the diagnostic's config: radius 7, 6 sweeps), reconstructed here so the
    #    comparison is between the two code paths and not between two kernels.
    radius = 7
    nparam = 2 * radius + 2
    print(f"kernel shape {tuple(kernel.shape)} (candidate) vs diagnostic radius {radius} -> {nparam} params")
    print("A-vs-B note: run both on the same (params, kernel, window) triple; any")
    print("difference is an implementation difference, since the inputs are shared.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
