"""CM-1: is the compiled metric the *right* metric?

The scored attention is `official_eval._attention`: a plain softmax over all
keys, NO causal mask.  With that, changing q_t changes only row t of the output:

    dO[t] = (1/sqrt(d)) dq_t^T G_t ,  G_t = K^T S_t^T V ,  S_t = diag(p_t) - p_t p_t^T
    ||dO[t]||^2 = dq_t^T M_t dq_t ,   M_t = (1/d) G_t G_t^T

so the metric to compile is Mbar = E_t[M_t] per KV group, and the deployable
rule picks codes against Mbar instead of against the operand's own squared
error.  This card checks that claim before any candidate is built.

It reports three things:

  1. an INDEPENDENT check of the derivation -- the measured row
     ||Attn(Q+dQ)-Attn(Q)||^2 at token t against the analytic
     (1/d)||dq_t^T G_t||^2 sum.  Ratio 1.000 means the derivation and the
     measurement agree.  This is independent of whether the mechanism helps.

  2. Spearman(PRED, TRUE) for the compiled metric, against two baselines:
       IDENTITY  sum_h ||dq_h||^2 / d               -- the operand-MSE objective
       DIAGONAL  sum_h dq_h^T diag(Mbar) dq_h / d   -- per-channel importance,
                                                        no coupling

  3. the pre-registered verdict: PASS iff spearman(Mbar, TRUE) >= 0.5 AND
     strictly greater than both baselines.

If Mbar does not beat both baselines, the mechanism is not new and the card is
NO_EFFECT by its own criterion.

Smoke first (seconds), then the full run:

    CM1_LAYERS=0 CM1_FIT_TOKENS=2 CM1_EVAL_TOKENS=1 CM1_SAMPLES=1 python cm1_correlation.py

CPU only, small shape, no shard, no candidate.
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

LAYERS = tuple(int(x) for x in os.environ.get("CM1_LAYERS", "0,1,5,8,15,22").split(","))
FIT_WINDOWS = (0, 1, 2)     # metric is built on these
EVAL_WINDOWS = (3, 4)       # measured on these -- held out from the metric
FIT_TOKENS = int(os.environ.get("CM1_FIT_TOKENS", "24"))
EVAL_TOKENS = int(os.environ.get("CM1_EVAL_TOKENS", "8"))
SAMPLES_PER_TOKEN = int(os.environ.get("CM1_SAMPLES", "3"))
TAKE_RATE = 0.15
SEED = 20260910
MIN_SPEARMAN = 0.5


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
    print(f"device={os.environ.get('CM1_DEVICE', 'auto')}", flush=True)
    torch.set_grad_enabled(False)
    solution = load("cm_solution", SOLUTION)

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
    width = q_heads * dim
    device = torch.device(os.environ.get("CM1_DEVICE", "cuda" if torch.cuda.is_available() else "cpu"))
    generator = torch.Generator(device=device).manual_seed(SEED)

    def attention(q, k, v):
        # Cast all three here; mixing a float32 q with a float64 k is how this
        # call failed once already.
        return v2._attention(
            q.to(torch.float32)[None],
            k.to(torch.float32)[None],
            v.to(torch.float32)[None],
            q_heads, kv_heads, dim,
        )[0]

    def dense(params, shape):
        return (
            v2.dequantize_hif4(v2._cpu_params(params), shape).to(torch.float64).to(device)
        )

    def moments(kd, vd, qd, tokens, out):
        """Accumulate this window's share of E_t[M_t] PER HEAD into `out`.

        Keeping the head axis separate costs memory but removes the largest
        approximation in the earlier version, which averaged over the four Q
        heads sharing a KV group before measuring anything.  The group-level
        metric is derived from this one, so a single run yields both.
        """

        weight = 1.0 / float(dim * len(tokens))
        for h in range(q_heads):
            g = h // group
            kc, vc = kd[:, g, :], vd[:, g, :]
            logits = qd[:, h, :] @ kc.transpose(-1, -2) / math.sqrt(dim)
            probs = torch.softmax(logits, dim=-1)
            for t in tokens:
                p = probs[t]
                s = torch.diag(p) - torch.outer(p, p)
                gmat = kc.transpose(-1, -2) @ s.transpose(-1, -2) @ vc
                out[h] += (gmat @ gmat.transpose(-1, -2)) * weight

    def gmat_for(kd, vd, qd, h, g, t):
        logits = qd[:, h, :] @ kd[:, g, :].transpose(-1, -2) / math.sqrt(dim)
        probs = torch.softmax(logits, dim=-1)
        p = probs[t]
        s = torch.diag(p) - torch.outer(p, p)
        return kd[:, g, :].transpose(-1, -2) @ s.transpose(-1, -2) @ vd[:, g, :]

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

        mbar = torch.zeros(q_heads, dim, dim, dtype=torch.float64, device=device)
        for index in FIT_WINDOWS:
            entry = windows[index]
            shapes = [v2.dequantize_nvfp4(*entry[r]).shape for r in ("q", "k", "v")]
            pq = solution.hif4_dynamic_quantize_q(*entry["q"], q_heads, dim, states["q_state"])
            pk = solution.hif4_dynamic_quantize_k(*entry["k"], kv_heads, dim, states["k_state"])
            pv = solution.hif4_dynamic_quantize_v(*entry["v"], kv_heads, dim, states["v_state"])
            moments(
                dense(pk, shapes[1]).reshape(-1, kv_heads, dim),
                dense(pv, shapes[2]).reshape(-1, kv_heads, dim),
                dense(pq, shapes[0]).reshape(-1, q_heads, dim),
                pick_tokens(int(shapes[0][0]), FIT_TOKENS),
                mbar,
            )

        # Group metric = mean of the group's per-head metrics (the earlier
        # version's only form); per-head is the less-averaged one.
        mbar_group = torch.stack(
            [mbar[g * group:(g + 1) * group].mean(0) for g in range(kv_heads)]
        )

        for w_index in EVAL_WINDOWS:
            entry = windows[w_index]
            shapes = [v2.dequantize_nvfp4(*entry[r]).shape for r in ("q", "k", "v")]
            pq = solution.hif4_dynamic_quantize_q(*entry["q"], q_heads, dim, states["q_state"])
            pk = solution.hif4_dynamic_quantize_k(*entry["k"], kv_heads, dim, states["k_state"])
            pv = solution.hif4_dynamic_quantize_v(*entry["v"], kv_heads, dim, states["v_state"])
            q0 = dense(pq, shapes[0])
            k0 = dense(pk, shapes[1])
            v0 = dense(pv, shapes[2])
            base = attention(q0, k0, v0)
            qd = q0.reshape(-1, q_heads, dim)
            kd = k0.reshape(-1, kv_heads, dim)
            vd = v0.reshape(-1, kv_heads, dim)

            step = (
                pq["scale_factor"].to(torch.float64)
                * pq["scale_lv2"].to(torch.float64)
                * pq["scale_lv3"].to(torch.float64)
                / 4.0
            ).expand_as(pq["mant"].to(torch.float64))
            sign = pq["sign"].to(torch.float64)
            mant = pq["mant"].to(torch.float64) * 4.0
            # The HiF4 param dict is a nested grid (tokens, 64, 8, 2, 4), not a
            # flat (tokens, width) tensor -- flatten before indexing by token.
            step_flat = (sign * step).reshape(-1, width).to(device)
            movable_flat = ((sign != 0) & (mant > 0) & (mant < 7)).reshape(-1, width).to(device)

            for t in pick_tokens(int(shapes[0][0]), EVAL_TOKENS):
                gcache = {
                    h: gmat_for(kd, vd, qd, h, h // group, t) for h in range(q_heads)
                }
                for _ in range(SAMPLES_PER_TOKEN):
                    direction = torch.where(
                        torch.rand(q_heads, dim, generator=generator, device=device) < 0.5, -1.0, 1.0
                    ).reshape(-1)
                    take = (
                        torch.rand(q_heads, dim, generator=generator, device=device) < TAKE_RATE
                    ).reshape(-1) & movable_flat[t]
                    if not bool(take.any()):
                        continue
                    dq_flat = torch.zeros_like(step_flat)
                    dq_flat[t] = torch.where(
                        take, direction * step_flat[t], torch.zeros_like(step_flat[t])
                    )
                    out = attention(q0 + dq_flat, k0, v0)
                    true = float((out[t] - base[t]).square().sum())

                    dq_row = dq_flat[t].reshape(q_heads, dim)
                    analytic = 0.0
                    pred_metric = pred_head = pred_diag = pred_ident = 0.0
                    for h in range(q_heads):
                        d = dq_row[h]
                        row = d @ gcache[h] / math.sqrt(dim)
                        analytic += float(row.square().sum())
                        mh = mbar[h]
                        mg = mbar_group[h // group]
                        pred_head += float(d @ mh @ d)
                        pred_metric += float(d @ mg @ d)
                        pred_diag += float(d @ torch.diag(torch.diagonal(mg)) @ d)
                        pred_ident += float(d @ d)
                    scale = 1.0 / float(dim)
                    rows.append(
                        {
                            "layer": layer,
                            "window": w_index,
                            "token": int(t),
                            "true": true,
                            "analytic": analytic,
                            "pred_head": pred_head * scale,
                            "pred_metric": pred_metric * scale,
                            "pred_diag": pred_diag * scale,
                            "pred_ident": pred_ident * scale,
                            "moved": int(take.sum()),
                        }
                    )
        print(f"[L{layer:>2}] samples so far {len(rows)}", flush=True)

    # --- independent check of the derivation, printed before anything else ---
    ratios = [r["analytic"] / r["true"] for r in rows if r["true"] > 0]
    check = {
        "mean_ratio": statistics.mean(ratios) if ratios else float("nan"),
        "min_ratio": min(ratios) if ratios else float("nan"),
        "max_ratio": max(ratios) if ratios else float("nan"),
        "n": len(ratios),
    }
    print()
    print(
        "INDEPENDENT CHECK  analytic (1/d)||dq G_t||^2  vs  measured row ||dO[t]||^2:"
    )
    print(
        f"  n={check['n']}  ratio mean={check['mean_ratio']:.6f}  "
        f"range=[{check['min_ratio']:.6f}, {check['max_ratio']:.6f}]   (1.000 = agree)"
    )

    if len(rows) < 8:
        print("fewer than 8 samples -- smoke run, no correlation summary")
        (HERE / "cm1-correlation.json").write_text(
            json.dumps({"smoke": True, "samples": len(rows), "rows": rows}, indent=2) + "\n",
            encoding="utf-8",
        )
        return 0

    trues = [r["true"] for r in rows]
    print()
    print(f"samples={len(rows)}  distinct TRUE values={len(set(trues))}")
    print(f"TRUE mean={statistics.mean(trues):.6e}")
    results = {}
    for name in ("pred_head", "pred_metric", "pred_diag", "pred_ident"):
        results[name] = spearman([r[name] for r in rows], trues)
        print(f"  spearman({name:>11}, TRUE) = {results[name]:+.4f}")
    verdict = (
        "PASS"
        if results["pred_metric"] >= MIN_SPEARMAN
        and results["pred_metric"] > results["pred_diag"]
        and results["pred_metric"] > results["pred_ident"]
        else "FAIL"
    )
    print()
    print(
        f"pre-registered criterion: spearman(Mbar, TRUE) >= {MIN_SPEARMAN} "
        f"AND > both baselines  ->  {verdict}"
    )
    if results["pred_metric"] <= results["pred_ident"]:
        print("  note: the compiled metric does NOT beat the operand-MSE objective.")

    (HERE / "cm1-correlation.json").write_text(
        json.dumps(
            {
                "layers": list(LAYERS),
                "fit_windows": list(FIT_WINDOWS),
                "eval_windows": list(EVAL_WINDOWS),
                "fit_tokens": FIT_TOKENS,
                "eval_tokens": EVAL_TOKENS,
                "samples_per_token": SAMPLES_PER_TOKEN,
                "samples": len(rows),
                "min_spearman": MIN_SPEARMAN,
                "independent_check": check,
                "spearman": results,
                "verdict": verdict,
                "rows": rows,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"wrote {HERE / 'cm1-correlation.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
