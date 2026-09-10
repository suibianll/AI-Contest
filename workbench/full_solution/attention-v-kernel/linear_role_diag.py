"""Where does our Linear encoder lose 6x on fc_up / proj / fc_gate?

The role attribution found the whole Linear gap sits in the three roles that
touch the 9216-dim MLP space, and that the PLAYER's relative output error there
is 0.75% against k's 0.124%.  Because the standard codec is relatively WORSE on
those roles than on k, the shortfall is ours, not theirs.

O = X W^T, so the output error has an activation part and a weight part.  This
splits the player's relative error into

    act_rel   ||X_hat - X_ref|| / ||X_ref||
    w_rel     ||W_hat - W_ref|| / ||W_ref||
    out_rel   ||X_hat W_hat^T - X_ref W_ref^T|| / ||X_ref W_ref^T||

for each role, alongside the same numbers for the STANDARD codec, so it is
visible which side we are losing on and whether the standard codec is losing
there too.
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
    root = load("lrd_root", ROOT / "solution.py")
    sys.path.insert(0, str(ROOT / "evaluator"))
    import official_eval as v2  # noqa: PLC0415

    pack = torch.load(
        ROOT / "artifacts/official_eval/cache/qwen3.5-4b-proxy-v2.pt",
        map_location="cpu", mmap=True, weights_only=False,
    )
    layer = int(os.environ.get("LRD_LAYER", "0"))
    roles = os.environ.get("LRD_ROLES", "proj,k,fc_up,fc_gate,q").split(",")
    # GPU by default: the .venv torch is cu124 and the panel runs on cuda.
    # Hard-coding cpu here made every run 10-30x slower than it needed to be.
    dev = torch.device(os.environ.get("LRD_DEVICE", "cuda" if torch.cuda.is_available() else "cpu"))
    print(f"device={dev}")

    print(f"layer={layer}")
    print(f"{'role':<9}{'shape':>13} | {'act_rel':>9}{'w_rel':>9}{'out_rel':>9} | "
          f"{'STD act':>9}{'STD w':>9}{'STD out':>9}")
    for role in roles:
        calib = [
            tuple(t.to(dev) for t in v2._pair(pack["calibration_activations"][role][s][layer].to(torch.float32)))
            for s in range(2)
        ]
        wq, ws = v2._pair(pack["weights"][layer][role].to(torch.float32))
        wq, ws = wq.to(dev), ws.to(dev)
        cal = root.hif4_calibration_and_quantize_weight(wq, ws, calib)
        wp = cal["weight_params"]
        state = cal["activation_state"]
        w_ref = v2.dequantize_nvfp4(wq, ws).to(torch.float64)
        w_hat = v2.dequantize_hif4(v2._cpu_params(wp), tuple(pack["weights"][layer][role].shape)).to(torch.float64).to(dev)
        w_std = v2.decode_standard_hif4(v2.encode_standard_hif4(w_ref)).to(torch.float64)

        # one test window for this activation role
        tensors = pack["test_activations"][role]
        idx = next((i for i, t in enumerate(tensors) if t is not None and t[layer] is not None and t[layer].shape[0] <= 256), None)
        if idx is None:
            print(f"{role:<9} no usable window")
            continue
        xq, xs = v2._pair(tensors[idx][layer].to(torch.float32))
        xq, xs = xq.to(dev), xs.to(dev)
        x_ref = v2.dequantize_nvfp4(xq, xs).to(torch.float64)
        xp = root.hif4_dynamic_quantize_activation(xq, xs, dict(state))
        x_hat = v2.dequantize_hif4(v2._cpu_params(xp), tuple(x_ref.shape)).to(torch.float64).to(dev)
        x_std = v2.decode_standard_hif4(v2.encode_standard_hif4(x_ref)).to(torch.float64)

        # The activation is encoded in a PERMUTED channel order and the weight's
        # columns are permuted to match, so X_hat @ W_hat^T is invariant while
        # ||X_hat - X_ref|| is not.  Compare against the reference taken through
        # the same permutation, or the operand numbers are meaningless (this is
        # what produced act_rel > 1 on the first run).
        perm = state.get("permutation") if isinstance(state, dict) else None
        if torch.is_tensor(perm):
            order = perm.detach().to(device=dev, dtype=torch.int64).reshape(-1)
            x_ref_cmp = x_ref.index_select(-1, order)
            w_ref_cmp = w_ref.index_select(-1, order)
        else:
            x_ref_cmp, w_ref_cmp = x_ref, w_ref

        def rel(a, b):
            return float((a - b).norm() / b.norm().clamp_min(1e-30))

        o_ref = x_ref @ w_ref.T
        # Mixed arms, ALL in our own coordinate: re-encode our decoded operands
        # with the standard codec, so the only thing that changes is codec
        # quality.  Mixing a standard operand with ours in the REFERENCE
        # coordinate would compare different bases -- that is what made
        # act_rel > 1 earlier.  Effective-bits note: standard quality is a
        # DEGRADATION here, so whichever side hurts more when replaced is where
        # our current advantage lives.
        x_std_ours = v2.decode_standard_hif4(v2.encode_standard_hif4(x_hat)).to(torch.float64)
        w_std_ours = v2.decode_standard_hif4(v2.encode_standard_hif4(w_hat)).to(torch.float64)
        arm_oo = rel(x_hat @ w_hat.T, o_ref)
        arm_so = rel(x_std_ours @ w_hat.T, o_ref)
        arm_os = rel(x_hat @ w_std_ours.T, o_ref)
        arm_ss = rel(x_std_ours @ w_std_ours.T, o_ref)
        print(
            f"{role:<9}{str(tuple(w_ref.shape)):>13} perm={'y' if torch.is_tensor(perm) else 'n'} | "
            f"oo={arm_oo:.4f}  so(aspect)={arm_so:.4f}  os(weight)={arm_os:.4f}  ss={arm_ss:.4f}  "
            f"| STDref={rel(x_std @ w_std.T, o_ref):.4f}"
        )
    print()
    print("act_rel/w_rel are operand-level; out_rel is what the score sees.")
    print("If OUR act_rel is the outlier, the activation quantizer is the weak side")
    print("for that role; if OUR w_rel is, it is the weight side.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
