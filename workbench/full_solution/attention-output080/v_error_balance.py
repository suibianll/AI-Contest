"""Error-balanced V rounding, measured on the panel objective.

The objective is the attention output MSE, and with the parent probabilities A

    O = A V ,  dO = A dV ,  ||dO||^2 = ||A dV||^2 ,

and A's rows sum to one.  So the part of dV that is constant across key tokens
passes through at full weight while the oscillating part largely cancels -- which
is why nearest-element rounding, whose errors are independent and therefore grow a
common mode, is the wrong rule for this objective.  Measured headroom: removing
that common mode by a value shift raises the panel gain from 0.5340 to 0.5751.

This script realises it as an actual code choice.  Per channel (fixed channel
index, all key tokens), starting from the parent's codes:

  * a legal +-1 step on `mant` moves that element's reconstruction by
    `sign * step`, so it moves the channel's summed error by the same amount;
  * greedily flip the element whose step best reduces `|sum_k delta[k]|` while
    adding the least operand error, repeat a bounded number of times.

Everything it reads is the V API's own input plus the state's frozen scales -- no
attention probabilities, no other API's input -- so it satisfies the deployment
constraint the plan sets.

Reported: the panel gain with the balanced codes, against the parent's, over the
72 attention cases, using the evaluator's own attention and reference.

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

MAX_ITER = 6
MAX_TOKENS = 256


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def balanced_params(p, x_ref, max_iter=MAX_ITER):
    """Greedy per-channel code choice minimising the summed per-channel error."""

    out = {k: v.clone() for k, v in p.items()}
    mant = out["mant"].float()
    sign = out["sign"].float()
    step = (
        out["scale_factor"].float() * out["scale_lv2"].float() * out["scale_lv3"].float() / 4
    ).expand_as(mant)
    m = mant * 4

    # Elements with sign 0 are frozen (a zero code carries no sign).
    movable = sign != 0
    value = sign * m * step
    delta = value - x_ref.reshape_as(value)
    flat_delta = delta.reshape(delta.shape[0], -1)
    flat_step = (sign * step).reshape(step.shape[0], -1)
    flat_m = m.reshape(m.shape[0], -1)
    flat_move = movable.reshape(movable.shape[0], -1)
    summed = flat_delta.sum(dim=0)
    operand = (flat_delta ** 2).sum(dim=0)

    flips = 0
    for _ in range(max_iter):
        # For each channel, the best single flip: choose the direction that
        # reduces |summed| and among those the smallest operand increase.
        target = -summed.unsqueeze(0)                       # what we want to add
        for direction in (1.0, -1.0):
            change = flat_step * direction
            legal = flat_move.clone()
            legal &= (flat_m + direction >= 0) & (flat_m + direction <= 7)
            new_sum = summed.unsqueeze(0) + change
            gain = summed.abs().unsqueeze(0) - new_sum.abs()
            cost = (flat_delta + change).square() - flat_delta.square()
            score = gain / cost.abs().clamp_min(1e-30)
            score = torch.where(legal & (gain > 0), score, torch.full_like(score, -1.0))
            best = score.argmax(dim=0)
            picked = score.gather(0, best.unsqueeze(0)).squeeze(0) > 0
            for channel in torch.nonzero(picked).reshape(-1).tolist():
                row = int(best[channel])
                if not bool(legal[row, channel]):
                    continue
                flat_m[row, channel] += direction
                flat_delta[row, channel] += flat_step[row, channel] * direction
                summed[channel] += flat_step[row, channel] * direction
                flips += 1

    out["mant"] = (flat_m.reshape(mant.shape) / 4).to(p["mant"].dtype)
    code = flat_m.reshape(mant.shape)
    new_sign = torch.where(code == 0, torch.zeros_like(sign), sign)
    out["sign"] = new_sign.to(p["sign"].dtype)
    return out, flips


def main() -> int:
    torch.set_num_threads(2)
    torch.set_grad_enabled(False)
    solution = load("eb_solution", ROOT / "solution.py")

    sys.path.insert(0, str(ROOT / "evaluator"))
    import official_eval as v2  # noqa: PLC0415

    pack = torch.load(
        ROOT / "artifacts/official_eval/cache/qwen3.5-4b-proxy-v2.pt",
        map_location="cpu", mmap=True, weights_only=False,
    )
    q_heads = int(pack["q_heads"])
    kv_heads = int(pack["kv_heads"])
    head_dim = int(pack["head_dim"])
    device = torch.device("cpu")
    layers = sorted({int(x) for x in pack["metadata"]["attention_layers"]})

    def attention(q, k, v):
        return v2._attention(*(t[None] for t in (q, k, v)), q_heads, kv_heads, head_dim)

    rows = []
    for layer in layers:
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
        states = solution.hif4_calibration_attention(windows, q_heads, kv_heads, head_dim)

        for window in range(len(pack["test_qkv"])):
            entry = pack["test_qkv"][window][layer]
            if entry is None:
                continue
            pairs = [v2._pair(entry[i].to(torch.float32)) for i in range(3)]
            if int(pairs[2][0].shape[0]) > MAX_TOKENS:
                continue
            refs = [v2.dequantize_nvfp4(*p).to(torch.float32) for p in pairs]
            standards = [
                v2.decode_standard_hif4(v2.encode_standard_hif4(r)).to(torch.float32)
                for r in refs
            ]
            pq = solution.hif4_dynamic_quantize_q(*pairs[0], q_heads, head_dim, states["q_state"])
            pk = solution.hif4_dynamic_quantize_k(*pairs[1], kv_heads, head_dim, states["k_state"])
            pv = solution.hif4_dynamic_quantize_v(*pairs[2], kv_heads, head_dim, states["v_state"])
            q_hat = v2.dequantize_hif4(v2._cpu_params(pq), refs[0].shape).to(torch.float32)
            k_hat = v2.dequantize_hif4(v2._cpu_params(pk), refs[1].shape).to(torch.float32)
            v_hat = v2.dequantize_hif4(v2._cpu_params(pv), refs[2].shape).to(torch.float32)

            bal, flips = balanced_params(pv, refs[2])
            v_bal = v2.dequantize_hif4(v2._cpu_params(bal), refs[2].shape).to(torch.float32)

            target = attention(refs[0], refs[1], refs[2])
            mse_std = float((attention(standards[0], standards[1], standards[2]) - target).square().mean())
            mse_cur = float((attention(q_hat, k_hat, v_hat) - target).square().mean())
            mse_bal = float((attention(q_hat, k_hat, v_bal) - target).square().mean())

            rows.append(
                {
                    "layer": layer,
                    "window": window,
                    "flips": flips,
                    "gain_current": 1.0 - mse_cur / mse_std,
                    "gain_balanced": 1.0 - mse_bal / mse_std,
                    "operand_mse_current": float((v_hat - refs[2]).square().mean()),
                    "operand_mse_balanced": float((v_bal - refs[2]).square().mean()),
                }
            )

    cur = statistics.mean(r["gain_current"] for r in rows)
    bal = statistics.mean(r["gain_balanced"] for r in rows)
    print(f"cases={len(rows)}")
    print(f"gain current  = {cur:.4f}")
    print(f"gain balanced = {bal:.4f}   ({bal - cur:+.4f}, target 0.80)")
    print(f"mean code flips per case = {statistics.mean(r['flips'] for r in rows):.0f}")
    print(
        f"operand MSE {statistics.mean(r['operand_mse_current'] for r in rows):.8f} -> "
        f"{statistics.mean(r['operand_mse_balanced'] for r in rows):.8f}"
    )
    per_layer = {}
    for r in rows:
        per_layer.setdefault(r["layer"], []).append(r)
    for layer, sub in sorted(per_layer.items()):
        print(
            f"  layer {layer:>2}: {statistics.mean(x['gain_current'] for x in sub):.4f} -> "
            f"{statistics.mean(x['gain_balanced'] for x in sub):.4f}"
        )

    (HERE / "v-error-balance.json").write_text(
        json.dumps({"cases": len(rows), "gain_current": cur, "gain_balanced": bal, "rows": rows}, indent=2)
        + "\n",
        encoding="utf-8",
    )
    print(f"wrote {HERE / 'v-error-balance.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
