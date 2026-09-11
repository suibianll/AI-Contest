"""UNFINISHED -- the frame map below is still WRONG, do not cite any output.

Three successive attempts at reproducing the player's operand frame all failed
the script's own check (mapX/mapW), and the guard correctly withheld the arms
each time.  The exact activation-side transform was then read off the source
(`solution.py:8779-8810` and `:11815-11827`, which agree): the order is

    dense = x * smooth_inv ; permute ; block-Hadamard ; + (dense @ U) @ V^T

so the earlier guesses were wrong in the scale direction (multiply, not divide)
and omitted the block Hadamard.  The weight side's inverse-transpose has not
been re-derived against the source yet.  This file is kept only so the failure
is on record; it has produced no usable reading.

--------------------------------------------------------------------------
Where does the Linear error actually live -- activation side or weight side?

The role attribution left fc_gate/fc_up/proj at ~0.48-0.52 gain while k sits at
0.82, and the two-sided swap arms (so ~= os) said neither side is individually
to blame.  This settles it in the OUTPUT metric with a coordinate-correct
four-arm decomposition.

WHY A COORDINATE MAP IS MANDATORY, AND HOW IT IS VALIDATED

The player does not hand the codec X_ref and W_ref.  The Linear calibration
learns a transform, applies one half to the activation and the exactly-inverse
half to the weight, so that only the PRODUCT survives:

    X_hat ~ quant(X_ours(X_ref))      W_hat ~ quant(W_ours(W_ref))
    X_ours(x) @ W_ours(w)^T  ==  x @ w^T     for every x, w

`la1 v1` fed X_hat and W_ref into one matmul -- two different bases -- and got
act-ours = -120, i.e. 120x worse than standard.  That absurdity is the signature
of a coordinate bug, not a measurement.

The map, read off the calibration source (`solution.py:8402-8432`, `:8540`):

    T_act = D (I + U V^T)      D = diag(1/smooth_inv)
    T_w   = D^-1 (I - V U^T)   <- forced, since T_act T_w^T = I requires it,
                                  and the code enforces V^T U ~ 0 for exactly
                                  this reason (the [L-R2] vtu_cross_max line)

    X_ours(x) = (x * 1/si) + ((x * 1/si) @ U) @ V^T
    W_ours(w) = (w * si)   - ((w * si)   @ V) @ U^T

This is a low-rank-perturbed diagonal -- NOT a plain diagonal.  Because any
T, T^-T pair satisfies the product identity, that identity CANNOT validate the
map (v1's assert passed while the answer was -120).  The validation used here is
the only one that can fail:

    X_hat must equal X_ours(X_ref) up to QUANTIZATION ERROR, and likewise W_hat.
    rel(X_hat, X_ours(X_ref)) and rel(W_hat, W_ours(W_ref)) must be small.

A wrong map makes those ratios O(1) or absurd, and the script aborts instead of
reporting a number.

Arms, all scored as MSE against ref = X_ref @ W_ref^T, reported as
gain = 1 - MSE_arm / MSE_STD:

    ours       X_hat          @ W_hat^T
    act-ours   X_hat          @ W_ours(W_ref)^T     weight error removed
    w-ours     X_ours(X_ref)  @ W_hat^T             activation error removed
    act-std    std(X_hat)      @ W_hat^T
    w-std      X_hat          @ std(W_hat)^T

Read: if act-ours ~= ours the weight has no headroom and the activation is the
whole gap; if w-ours ~= ours the activation is clean and the weight is.  If BOTH
sit well above ours the two errors are cancelling, and no single-side fix works.

gain(std) is identically 0 -- it IS the denominator -- so STD is reported in
out_rel units instead.
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
    root = load("la1_root", ROOT / "solution.py")
    sys.path.insert(0, str(ROOT / "evaluator"))
    import official_eval as v2  # noqa: PLC0415

    pack = torch.load(
        ROOT / "artifacts/official_eval/cache/qwen3.5-4b-proxy-v2.pt",
        map_location="cpu", mmap=True, weights_only=False,
    )
    dev = torch.device(os.environ.get("LA1_DEVICE", "cuda" if torch.cuda.is_available() else "cpu"))
    layer = int(os.environ.get("LA1_LAYER", "0"))
    roles = os.environ.get("LA1_ROLES", "fc_gate,fc_up,proj,q,k").split(",")
    print(f"device={dev} layer={layer}", flush=True)

    def dec_h(params, shape):
        return v2.dequantize_hif4(v2._cpu_params(params), tuple(shape)).to(torch.float64).to(dev)

    def rel(a, b):
        return float((a - b).norm() / b.norm().clamp_min(1e-30))

    def frames(state, dev):
        """Return (x_ours, w_ours): maps reference-frame operands into our frame."""
        get = (lambda k: state.get(k)) if isinstance(state, dict) else (lambda k: None)
        si = get("smooth_inv")
        u = get("residual_u")
        v = get("residual_v")
        perm = get("permutation")
        f64 = lambda t: t.to(device=dev, dtype=torch.float64)  # noqa: E731
        si = f64(si).reshape(-1) if torch.is_tensor(si) else None
        u = f64(u) if torch.is_tensor(u) else None
        v = f64(v) if torch.is_tensor(v) else None
        order = f64(perm).to(torch.int64).reshape(-1) if torch.is_tensor(perm) else None

        def x_ours(x):
            y = x if si is None else x * (1.0 / si)
            if u is not None and v is not None:
                y = y + (y @ u) @ v.t()
            return y if order is None else y.index_select(-1, order)

        def w_ours(w):
            y = w if si is None else w * si
            if u is not None and v is not None:
                y = y - (y @ v) @ u.t()
            return y if order is None else y.index_select(-1, order)

        return x_ours, w_ours

    print(f"{'role':<9}{'n':>5}{'in':>7} | {'ours':>8}{'act-ours':>9}{'w-ours':>8}"
          f" | {'act-std':>8}{'w-std':>8} | {'STDout':>8} | {'mapX':>7}{'mapW':>7}")
    for role in roles:
        calib = [
            tuple(t.to(dev) for t in v2._pair(pack["calibration_activations"][role][s][layer].to(torch.float32)))
            for s in range(2)
        ]
        wq, ws = v2._pair(pack["weights"][layer][role].to(torch.float32))
        wq, ws = wq.to(dev), ws.to(dev)
        cal = root.hif4_calibration_and_quantize_weight(wq, ws, calib)
        state = cal["activation_state"]
        x_ours, w_ours = frames(state, dev)

        w_ref = v2.dequantize_nvfp4(wq, ws).to(torch.float64)
        w_hat = dec_h(cal["weight_params"], w_ref.shape)
        w_std = v2.decode_standard_hif4(v2.encode_standard_hif4(w_ref)).to(torch.float64)

        tensors = pack["test_activations"][role]
        idx = next(
            (i for i, t in enumerate(tensors)
             if t is not None and t[layer] is not None and t[layer].shape[0] <= 256),
            None,
        )
        if idx is None:
            print(f"{role:<9} no usable window")
            continue
        xq, xs = v2._pair(tensors[idx][layer].to(torch.float32))
        xq, xs = xq.to(dev), xs.to(dev)
        x_ref = v2.dequantize_nvfp4(xq, xs).to(torch.float64)
        xp = root.hif4_dynamic_quantize_activation(xq, xs, dict(state))
        x_hat = dec_h(xp, x_ref.shape)
        x_std = v2.decode_standard_hif4(v2.encode_standard_hif4(x_ref)).to(torch.float64)

        # THE validation.  A wrong map cannot make these small.
        map_x = rel(x_hat, x_ours(x_ref))
        map_w = rel(w_hat, w_ours(w_ref))
        if not (map_x < 0.5 and map_w < 0.5):
            print(f"{role:<9} MAP REJECTED: rel(X_hat, X_ours(X_ref))={map_x:.4f} "
                  f"rel(W_hat, W_ours(W_ref))={map_w:.4f} -- not a quantization-level "
                  f"gap, so the frame is wrong. Refusing to report arms.", flush=True)
            continue

        x_ref_o, w_ref_o = x_ours(x_ref), w_ours(w_ref)
        ref = x_ref @ w_ref.T
        invar = rel(x_ref_o @ w_ref_o.T, ref)
        assert invar < 1e-9, f"our two half-maps are not inverse: {invar}"

        def mse(a, b):
            return float((a - b).square().mean())

        base = mse(x_std @ w_std.T, ref)
        arms = {
            "ours": x_hat @ w_hat.T,
            "act-ours": x_hat @ w_ref_o.T,
            "w-ours": x_ref_o @ w_hat.T,
            "act-std": v2.decode_standard_hif4(v2.encode_standard_hif4(x_hat)).to(torch.float64) @ w_hat.T,
            "w-std": x_hat @ v2.decode_standard_hif4(v2.encode_standard_hif4(w_hat)).to(torch.float64).T,
        }
        gains = {k: 1.0 - mse(v, ref) / max(base, 1e-30) for k, v in arms.items()}
        print(
            f"{role:<9}{int(x_ref.shape[0]):>5}{int(x_ref.shape[1]):>7} | "
            f"{gains['ours']:>8.4f}{gains['act-ours']:>9.4f}{gains['w-ours']:>8.4f} | "
            f"{gains['act-std']:>8.4f}{gains['w-std']:>8.4f} | "
            f"{rel(x_std @ w_std.T, ref):>8.4f} | {map_x:>7.4f}{map_w:>7.4f}",
            flush=True,
        )
    print()
    print("mapX/mapW are the frame check: rel(X_hat, X_ours(X_ref)) and rel(W_hat, W_ours(W_ref)).")
    print("They must sit at quantization-error level; if they do not, the arms are withheld.")
    print("act-ours ~= ours  -> the weight has no headroom; the activation is the whole gap")
    print("w-ours   ~= ours  -> the activation is clean; the weight is the whole gap")
    print("both >> ours      -> the two errors cancel; only a JOINT (A,W) move helps")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
