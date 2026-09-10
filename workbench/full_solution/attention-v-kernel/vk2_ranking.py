"""VK-2: does the position kernel RANK V code choices the way A does?

VK-1 measured how far the Toeplitz kernel W is from A in *magnitude* on the real
residual, and found ||(W-A)dV|| / ||A dV|| ~ 0.80.  That number does not settle
whether W can drive a rule, because a rule uses W to *rank* alternatives, not to
predict an absolute error: a surrogate can be badly mis-scaled and still order
choices the same way.

So this measures exactly that.  For legal +-1 mantissa corrections of V's codes,

    A_true   sum_h sum_t ||(A_h d_h)[t]||^2      the quantity the score contains
    W_pred   sum_h sum_t ||(W_h d_h)[t]||^2      W[t,k] = w[k-t], fitted on FIT windows
    U_pred   sum_h T ||mean_k d_h[k]||^2         the uniform kernel (v_error_balance)
    I_pred   sum_h ||d_h||^2                     operand MSE

and reports the within-window Spearman of each predictor against A_true.  A rule
driven by a predictor can only be as good as that predictor's ordering.

W's rows are renormalised to unit sum: the least-squares fit minimises entrywise
error and does not force row-stochasticity, while the true A is row-stochastic,
so leaving the scale in would compare scales rather than orderings.

No threshold is applied and no verdict is declared (plan section 6.1).

Smoke:  VK2_LAYERS=0 VK2_SAMPLES=4 python vk2_ranking.py
"""

from __future__ import annotations

import importlib.util
import json
import math
import os
import statistics
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
SOLUTION = ROOT / "solution.py"

LAYERS = tuple(int(x) for x in os.environ.get("VK2_LAYERS", "0,1,5,8,15,22").split(","))
FIT_WINDOWS = (0, 1, 2)
EVAL_WINDOWS = (3, 4)
FIT_TOKENS = int(os.environ.get("VK2_FIT_TOKENS", "48"))
SAMPLES = int(os.environ.get("VK2_SAMPLES", "16"))
KERNEL_RADIUS = 15
TAKE_RATE = 0.15
TOKEN_RATE = 0.20       # fraction of tokens perturbed per sample
SEED = 20260911


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def spearman(xs, ys):
    def ranks(values):
        order = sorted(range(len(values)), key=lambda i: values[i])
        out = [0.0] * len(values)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
                j += 1
            for k in range(i, j + 1):
                out[order[k]] = (i + j) / 2.0
            i = j + 1
        return out

    rx, ry = ranks(list(xs)), ranks(list(ys))
    n = len(rx)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    dx = math.sqrt(sum((a - mx) ** 2 for a in rx))
    dy = math.sqrt(sum((b - my) ** 2 for b in ry))
    return num / (dx * dy) if dx > 0 and dy > 0 else float("nan")


def pick_tokens(total: int, count: int) -> list:
    if total <= count:
        return list(range(total))
    return sorted({int(round(x)) for x in torch.linspace(0, total - 1, count).tolist()})


def main() -> int:
    torch.set_num_threads(4)
    torch.set_grad_enabled(False)
    solution = load("vk2_solution", SOLUTION)

    sys.path.insert(0, str(ROOT / "evaluator"))
    import official_eval as v2  # noqa: PLC0415

    pack = torch.load(
        ROOT / "artifacts/official_eval/cache/qwen3.5-4b-proxy-v2.pt",
        map_location="cpu", mmap=True, weights_only=False,
    )
    q_heads = int(pack["q_heads"])
    kv_heads = int(pack["kv_heads"])
    dim = int(pack["head_dim"])
    group = q_heads // kv_heads
    device = torch.device(os.environ.get("VK2_DEVICE", "cuda" if torch.cuda.is_available() else "cpu"))
    generator = torch.Generator(device=device).manual_seed(SEED)
    print(f"device={device}", flush=True)

    def dense(params, shape):
        return v2.dequantize_hif4(v2._cpu_params(params), shape).to(torch.float64).to(device)

    def probs_for(qd, kd, h, g):
        logits = qd[:, h, :] @ kd[:, g, :].transpose(-1, -2) / math.sqrt(dim)
        return torch.softmax(logits, dim=-1)

    def bucket_ids(tokens):
        idx = torch.full((tokens, tokens), 2 * KERNEL_RADIUS, dtype=torch.long)
        for t in range(tokens):
            for k in range(tokens):
                off = k - t
                if abs(off) <= KERNEL_RADIUS:
                    idx[t, k] = off + KERNEL_RADIUS
        return idx

    rows = []
    for layer in LAYERS:
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
        states = solution.hif4_calibration_attention(windows, q_heads, kv_heads, dim)

        totals = torch.zeros(q_heads, 2 * KERNEL_RADIUS + 1, dtype=torch.float64, device=device)
        counts = torch.zeros(2 * KERNEL_RADIUS + 1, dtype=torch.float64, device=device)
        for index in FIT_WINDOWS:
            entry = windows[index]
            shapes = [v2.dequantize_nvfp4(*entry[r]).shape for r in ("q", "k", "v")]
            pq = solution.hif4_dynamic_quantize_q(*entry["q"], q_heads, dim, states["q_state"])
            pk = solution.hif4_dynamic_quantize_k(*entry["k"], kv_heads, dim, states["k_state"])
            qd = dense(pq, shapes[0]).reshape(-1, q_heads, dim)
            kd = dense(pk, shapes[1]).reshape(-1, kv_heads, dim)
            keep = pick_tokens(int(shapes[0][0]), FIT_TOKENS)
            idx = bucket_ids(int(shapes[0][0])).to(device)
            for h in range(q_heads):
                p = probs_for(qd, kd, h, h // group)[keep]
                sub = idx[keep]
                for b in range(2 * KERNEL_RADIUS + 1):
                    sel = sub == b
                    if bool(sel.any()):
                        totals[h, b] += p[sel].sum()
                        if h == 0:
                            counts[b] += float(sel.sum())
        w = totals / counts.clamp_min(1.0)[None, :]

        for w_index in EVAL_WINDOWS:
            entry = windows[w_index]
            shapes = [v2.dequantize_nvfp4(*entry[r]).shape for r in ("q", "k", "v")]
            pq = solution.hif4_dynamic_quantize_q(*entry["q"], q_heads, dim, states["q_state"])
            pk = solution.hif4_dynamic_quantize_k(*entry["k"], kv_heads, dim, states["k_state"])
            pv = solution.hif4_dynamic_quantize_v(*entry["v"], kv_heads, dim, states["v_state"])
            qd = dense(pq, shapes[0]).reshape(-1, q_heads, dim)
            kd = dense(pk, shapes[1]).reshape(-1, kv_heads, dim)
            tokens = int(shapes[0][0])
            idx = bucket_ids(tokens).to(device)
            a_by_head = [probs_for(qd, kd, h, h // group) for h in range(q_heads)]
            wk_by_head = [w[h][idx] for h in range(q_heads)]
            wk_by_head = [x / x.sum(-1, keepdim=True).clamp_min(1e-30) for x in wk_by_head]

            step = (
                pv["scale_factor"].to(torch.float64)
                * pv["scale_lv2"].to(torch.float64)
                * pv["scale_lv3"].to(torch.float64)
                / 4.0
            ).expand_as(pv["mant"].to(torch.float64))
            sign = pv["sign"].to(torch.float64)
            mant = pv["mant"].to(torch.float64) * 4.0
            width = kv_heads * dim
            step_flat = (sign * step).reshape(-1, width).to(device)
            movable = ((sign != 0) & (mant > 0) & (mant < 7)).reshape(-1, width).to(device)

            step3 = step_flat.reshape(tokens, kv_heads, dim)
            movable3 = movable.reshape(tokens, kv_heads, dim)
            trues, preds_w, preds_u, preds_i, touched = [], [], [], [], []
            for _ in range(SAMPLES):
                # MULTI-token perturbations.  A single-token perturbation makes
                # every predictor collapse onto ||delta||^2: with all mass at one
                # token, the uniform kernel gives T*||delta_k/T||^2 = ||delta_k||^2/T
                # (identical to the operand predictor up to a constant), and A_true
                # degenerates to (sum_t A[t,k]^2)||delta_k||^2 as well.  The probe
                # then measures nothing about the weighting shape it is asking about.
                tok_take = (
                    torch.rand(tokens, generator=generator, device=device) < TOKEN_RATE
                )[:, None, None]
                chan_take = (
                    torch.rand(tokens, kv_heads, dim, generator=generator, device=device) < TAKE_RATE
                )
                direction = torch.where(
                    torch.rand(tokens, kv_heads, dim, generator=generator, device=device) < 0.5,
                    -1.0, 1.0,
                )
                sel = tok_take & chan_take & movable3
                if int(sel.sum()) < 8:
                    continue
                d3 = torch.where(sel, direction * step3, torch.zeros_like(step3))
                touched.append(int(sel.any(-1).any(-1).sum()))

                a_true = w_pred = u_pred = i_pred = 0.0
                for h in range(q_heads):
                    g = h // group
                    dv = d3[:, g, :]
                    a_true += float((a_by_head[h] @ dv).square().sum())
                    w_pred += float((wk_by_head[h] @ dv).square().sum())
                    u_pred += tokens * float(dv.mean(0).square().sum())
                    i_pred += float(dv.square().sum())
                trues.append(a_true)
                preds_w.append(w_pred)
                preds_u.append(u_pred)
                preds_i.append(i_pred)

            if len(trues) < 4:
                continue
            rows.append(
                {
                    "layer": layer,
                    "window": w_index,
                    "tokens": tokens,
                    "n": len(trues),
                    "tokens_touched_mean": statistics.mean(touched) if touched else 0.0,
                    "rho_w": spearman(preds_w, trues),
                    "rho_u": spearman(preds_u, trues),
                    "rho_i": spearman(preds_i, trues),
                    # Magnitude-controlled reading: dividing both sides by the
                    # operand-MSE predictor removes the part of A_true that is
                    # just "how big is this perturbation", leaving the part that
                    # depends on the weighting SHAPE -- which is the only thing
                    # the kernel could contribute.
                    "rho_w_shape": spearman(
                        [a / b for a, b in zip(preds_w, preds_i)],
                        [a / b for a, b in zip(trues, preds_i)],
                    ),
                    "rho_u_shape": spearman(
                        [a / b for a, b in zip(preds_u, preds_i)],
                        [a / b for a, b in zip(trues, preds_i)],
                    ),
                    "raw": {
                        "a_true": trues, "w_pred": preds_w,
                        "u_pred": preds_u, "i_pred": preds_i,
                    },
                }
            )
            print(
                f"  [L{layer:>2}/w{w_index} T={tokens:>4} n={len(trues):>2}] "
                f"rho(W,A)={rows[-1]['rho_w']:+.3f}  rho(U,A)={rows[-1]['rho_u']:+.3f}  "
                f"rho(I,A)={rows[-1]['rho_i']:+.3f}  tok~{rows[-1]['tokens_touched_mean']:.0f}",
                flush=True,
            )

    if not rows:
        print("no samples")
        return 1

    print()
    print("within-window Spearman against A_true (1.000 = same ordering as the scored A):")
    for key, name in (("rho_w", "KERNEL "), ("rho_u", "UNIFORM"), ("rho_i", "OPERAND")):
        v = [r[key] for r in rows]
        print(f"  {name}  mean={statistics.mean(v):+.4f}  range=[{min(v):+.4f}, {max(v):+.4f}]  n={len(v)}")
    print()
    print("magnitude-controlled (shape only) -- does the weighting add anything")
    print("beyond how big the perturbation is?  Spearman(pred/operand, A_true/operand):")
    for key, name in (("rho_w_shape", "KERNEL "), ("rho_u_shape", "UNIFORM")):
        v = [r[key] for r in rows if not math.isnan(r[key])]
        print(f"  {name}  mean={statistics.mean(v):+.4f}  range=[{min(v):+.4f}, {max(v):+.4f}]  n={len(v)}")
    print()
    print("NO THRESHOLD IS APPLIED (plan section 6.1).")

    (HERE / "vk2-ranking.json").write_text(
        json.dumps(
            {"layers": list(LAYERS), "samples_per_window": SAMPLES, "rows": rows}, indent=2
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"wrote {HERE / 'vk2-ranking.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
