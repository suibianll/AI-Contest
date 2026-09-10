"""Linear side of the same family: how much of X's error can W annihilate?

The scored quantity for the Linear problem is the matmul output

    O = X W^T ,   O_hat = X_hat W_hat^T ,   gain = 1 - MSE(O_hat, O_ref)/MSE(O_std, O_ref)

so X's encoding error propagates as `delta_X W_hat^T`.  A row of that product is
`delta_X[o,:] . W_hat[o,:]`, which vanishes for every output o exactly when delta_X
lies in the **orthogonal complement of W_hat's row space**.  That complement has
dimension `in - rank(W_hat)`, so it is non-trivial precisely when the contraction
dimension exceeds the output dimension.

The same caveat as the V measurement applies: moving X by a projection is a *value*
shift, not a legal code change, so this measures the ceiling of the mechanism, not
the mechanism.  It is still the number that decides whether an encoder-side rule
aimed at that complement is worth building.

Two roles are measured for contrast, chosen by that dimension condition:

  q   in 2560 / out 4096   -> row space spans everything, complement trivial
  o   in 4096 / out 2560   -> complement at least 1536-dimensional

CPU only, no shard, no candidate.
"""

from __future__ import annotations

import importlib.util
import json
import statistics
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
    torch.set_num_threads(2)
    torch.set_grad_enabled(False)
    solution = load("ln_solution", ROOT / "solution.py")

    sys.path.insert(0, str(ROOT / "evaluator"))
    import official_eval as v2  # noqa: PLC0415

    pack = torch.load(
        ROOT / "artifacts/official_eval/cache/qwen3.5-4b-proxy-v2.pt",
        map_location="cpu", mmap=True, weights_only=False,
    )
    device = torch.device("cpu")
    layer = 0

    rows = []
    for role in ("q", "o"):
        calib = [
            tuple(t.to(device) for t in v2._pair(pack["calibration_activations"][role][s][layer].to(torch.float32)))
            for s in range(2)
        ]
        wq, ws = v2._pair(pack["weights"][layer][role].to(torch.float32))
        cal = solution.hif4_calibration_and_quantize_weight(
            wq.to(device), ws.to(device), calib
        )
        state = cal["activation_state"]
        w_hat = v2.dequantize_hif4(v2._cpu_params(cal["weight_params"]), tuple(pack["weights"][layer][role].shape)).to(torch.float32)
        w_ref = v2.dequantize_nvfp4(*[t.to(device) for t in v2._pair(pack["weights"][layer][role].to(torch.float32))]).to(torch.float32)
        w_std = v2.decode_standard_hif4(v2.encode_standard_hif4(w_ref)).to(torch.float32)

        for window in range(len(pack["test_activations"][role])):
            tensor = pack["test_activations"][role][window][layer]
            if tensor is None or tensor.shape[0] > 256:
                continue
            pair = v2._pair(tensor.to(torch.float32))
            x_ref = v2.dequantize_nvfp4(*[t.to(device) for t in pair]).to(torch.float32)
            x_std = v2.decode_standard_hif4(v2.encode_standard_hif4(x_ref)).to(torch.float32)
            x_pr = solution.hif4_dynamic_quantize_activation(pair[0].to(device), pair[1].to(device), dict(state))
            x_hat = v2.dequantize_hif4(v2._cpu_params(x_pr), x_ref.shape).to(torch.float32)

            o_ref = x_ref @ w_ref.T
            mse_std = float((x_std @ w_std.T - o_ref).square().mean())
            mse_cur = float((x_hat @ w_hat.T - o_ref).square().mean())

            # Project delta_X onto the row space of W_hat and remove it.
            delta = x_hat - x_ref
            gram = w_hat @ w_hat.T                       # (out, out)
            pinv = torch.linalg.pinv(gram)
            projector = w_hat.T @ pinv @ w_hat           # (in, in), onto rowspace(W_hat)
            x_bal = x_hat - delta @ projector
            mse_bal = float((x_bal @ w_hat.T - o_ref).square().mean())

            rank = int(torch.linalg.matrix_rank(w_hat))
            rows.append(
                {
                    "role": role,
                    "window": window,
                    "in_features": int(x_ref.shape[1]),
                    "out_features": int(w_hat.shape[0]),
                    "rank_w": rank,
                    "complement_dim": int(x_ref.shape[1]) - rank,
                    "gain_current": 1.0 - mse_cur / mse_std,
                    "gain_nullspace": 1.0 - mse_bal / mse_std,
                }
            )

    print(f"cases={len(rows)}")
    for role in ("q", "o"):
        sub = [r for r in rows if r["role"] == role]
        if not sub:
            continue
        cur = statistics.mean(r["gain_current"] for r in sub)
        bal = statistics.mean(r["gain_nullspace"] for r in sub)
        first = sub[0]
        print(
            f"  {role}: in={first['in_features']} out={first['out_features']} "
            f"rank(W)={first['rank_w']} complement={first['complement_dim']} | "
            f"gain {cur:.4f} -> {bal:.4f} ({bal - cur:+.4f})  n={len(sub)}"
        )

    (HERE / "linear-nullspace.json").write_text(
        json.dumps({"cases": len(rows), "rows": rows}, indent=2) + "\n", encoding="utf-8"
    )
    print(f"\nwrote {HERE / 'linear-nullspace.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
