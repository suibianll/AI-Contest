"""LK-1 prerequisite: does the block-diagonal part of `h` keep the mechanism's effect?

L-EM2 stores `h = W_hat^T W_hat - W_ref^T W_hat`, an in x in dense matrix, and the
dynamic API builds h_inv from it.  `_EM1_MAX_CHANNELS = 4096` excludes proj's
9216 because that inverse is ~12 s/layer there.

A block-diagonal h (per 64-channel block) would cut storage 140x and the inverse
20000x, but only if it keeps the effect.  Instead of reimplementing the descent,
this swaps `h` inside the state and calls the SAME dynamic API:

    ARM C (parent)   state with em1 removed          -> no descent
    ARM A (full)     the compiled state              -> full h
    ARM B (block)    same state, h replaced by its per-64-block diagonal

and compares output-space relative error, which this project has already shown
reproduces the panel for these roles.

If B ~= A the block approximation keeps the effect and LK-1 is worth building for
9216.  If B ~= C the effect lives in the off-block terms and LK-1 should be
cancelled.
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
    root = load("lk1_root", ROOT / "solution.py")
    sys.path.insert(0, str(ROOT / "evaluator"))
    import official_eval as v2  # noqa: PLC0415

    pack = torch.load(
        ROOT / "artifacts/official_eval/cache/qwen3.5-4b-proxy-v2.pt",
        map_location="cpu", mmap=True, weights_only=False,
    )
    dev = torch.device(os.environ.get("LK1_DEVICE", "cuda" if torch.cuda.is_available() else "cpu"))
    layer = int(os.environ.get("LK1_LAYER", "0"))
    roles = os.environ.get("LK1_ROLES", "q,k,fc_up,fc_gate,proj").split(",")
    print(f"device={dev} layer={layer}", flush=True)

    for role in roles:
        calib = [
            tuple(t.to(dev) for t in v2._pair(pack["calibration_activations"][role][s][layer].to(torch.float32)))
            for s in range(2)
        ]
        wq, ws = v2._pair(pack["weights"][layer][role].to(torch.float32))
        wq, ws = wq.to(dev), ws.to(dev)
        cal = root.hif4_calibration_and_quantize_weight(wq, ws, calib)
        state = cal["activation_state"]
        em1 = state.get("em1")
        if not isinstance(em1, dict) or not torch.is_tensor(em1.get("h")):
            print(f"{role:<9} em1={em1.get('arm') if isinstance(em1, dict) else 'absent'} -- no full h, skipping")
            continue
        h = em1["h"]
        n = int(h.shape[0])
        hd = torch.zeros_like(h)
        for b in range(0, n, 64):
            hd[b:b + 64, b:b + 64] = h[b:b + 64, b:b + 64]
        off_norm = float((h - hd).norm() / h.norm().clamp_min(1e-30))

        state_bd = dict(state)
        state_bd["em1"] = dict(em1)
        state_bd["em1"]["h"] = hd
        state_no = dict(state)
        state_no.pop("em1", None)

        idx = next(
            (i for i, t in enumerate(pack["test_activations"][role])
             if t is not None and t[layer] is not None and t[layer].shape[0] <= 256),
            None,
        )
        if idx is None:
            print(f"{role:<9} no usable test window")
            continue
        xq, xs = v2._pair(pack["test_activations"][role][idx][layer].to(torch.float32))
        xq, xs = xq.to(dev), xs.to(dev)
        x_ref = v2.dequantize_nvfp4(xq, xs).to(torch.float64)
        w_ref = v2.dequantize_nvfp4(wq, ws).to(torch.float64)
        o_ref = x_ref @ w_ref.T

        def out_rel(st):
            xp = root.hif4_dynamic_quantize_activation(xq, xs, st)
            x_hat = v2.dequantize_hif4(v2._cpu_params(xp), tuple(x_ref.shape)).to(torch.float64).to(dev)
            o = x_hat @ v2.dequantize_hif4(
                v2._cpu_params(cal["weight_params"]), tuple(w_ref.shape)
            ).to(torch.float64).to(dev).T
            return float((o - o_ref).norm() / o_ref.norm().clamp_min(1e-30))

        rc, ra, rb = out_rel(state_no), out_rel(state), out_rel(state_bd)
        print(
            f"{role:<9} in={n:>5} off-block energy={off_norm:.4f} | "
            f"C(parent)={rc:.4f}  A(full)={ra:.4f}  B(block)={rb:.4f}  "
            f"| A gain={rc - ra:+.5f}  B gain={rc - rb:+.5f}  B/A={(rc - rb) / max(rc - ra, 1e-12):.3f}",
            flush=True,
        )
    print()
    print("If B/A is near 1 the block-diagonal h keeps the effect; near 0 it does not.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
