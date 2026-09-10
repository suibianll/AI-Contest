"""VK-1 diagnostic: how well does a relative-position kernel stand in for A?

With Q and K held at their deployed values, the V-side output error is exact:

    dO[t] = sum_k A[t,k] * dV[k] ,   A = softmax(Q K^T / sqrt(d)),  dV = V_hat - V_ref
    ||dO||^2 = sum_t ||(A dV)[t]||^2

so the question this card asks is narrow and measurable: **if a deployable rule
may only use a kernel that depends on relative position, how much does it lose
against the true A -- and how much does it gain against the uniform kernel that
`v_error_balance.py` used?**

It reports, on real calibration windows, for the *actual* V residual:

    TRUE        sum_t ||(A dV)[t]||^2           through the real attention forward
    TRUE_A      the same quantity recomputed from the deployed A in float64
                -- an INDEPENDENT check: these two must agree, and if they do
                not, the A being used is not the attention being scored
    KERNEL      sum_t ||(W dV)[t]||^2           W[t,k] = w[k-t], w fitted on FIT windows
    UNIFORM     sum_t ||(1/T * sum_k dV[k])||^2 the unweighted proxy

The per-window ratio of each predictor to TRUE is the reading.  Per the plan
(`proposals/2026-09-10-attention-080-algorithm-research-next.md` section 6.1)
**no threshold is applied and no pass/fail is declared** -- these numbers only
describe the fixed-kernel approximation on the sampled windows.

Kernel: two-sided, |k-t| <= KERNEL_RADIUS, plus one shared bucket for larger
offsets.  A is NOT causal (the evaluator's `_attention` has no mask) and the
softmax is not guaranteed to decay with distance, so the fit is free to put mass
on either side; that is a property to be read off the fitted kernel, not assumed.

CPU or GPU; smoke first:

    CM1_DEVICE=cpu VK1_LAYERS=0 VK1_TOKENS=4 python vk1_diagnostic.py
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

LAYERS = tuple(int(x) for x in os.environ.get("VK1_LAYERS", "0,1,5,8,15,22").split(","))
FIT_WINDOWS = (0, 1, 2)
EVAL_WINDOWS = (3, 4)
FIT_TOKENS = int(os.environ.get("VK1_TOKENS", "48"))
KERNEL_RADIUS = 15          # |k-t| <= 15, plus one bucket beyond -> 32 parameters
SEED = 20260911


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def pick_tokens(total: int, count: int) -> list:
    if total <= count:
        return list(range(total))
    return sorted({int(round(x)) for x in torch.linspace(0, total - 1, count).tolist()})


def main() -> int:
    torch.set_num_threads(4)
    torch.set_grad_enabled(False)
    solution = load("vk_solution", SOLUTION)

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
    device = torch.device(os.environ.get("VK1_DEVICE", "cuda" if torch.cuda.is_available() else "cpu"))
    print(f"device={device}", flush=True)

    def dense(params, shape):
        return v2.dequantize_hif4(v2._cpu_params(params), shape).to(torch.float64).to(device)

    def attention(q, k, v):
        return v2._attention(
            q.to(torch.float32)[None],
            k.to(torch.float32)[None],
            v.to(torch.float32)[None],
            q_heads, kv_heads, dim,
        )[0]

    def probs_for(qd, kd, h, g):
        logits = qd[:, h, :] @ kd[:, g, :].transpose(-1, -2) / math.sqrt(dim)
        return torch.softmax(logits, dim=-1)

    def bucket_ids(tokens):
        """(t, k) -> kernel index, in [0, 2*RADIUS] plus the far bucket."""

        idx = torch.full((tokens, tokens), 2 * KERNEL_RADIUS, dtype=torch.long)
        for t in range(tokens):
            for k in range(tokens):
                offset = k - t
                if abs(offset) <= KERNEL_RADIUS:
                    idx[t, k] = offset + KERNEL_RADIUS
        return idx

    rows = []
    kernel_report = {}
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

        # --- fit the kernel: w_bucket = mean of A over the entries in it
        totals = torch.zeros(q_heads, 2 * KERNEL_RADIUS + 1, dtype=torch.float64, device=device)
        counts = torch.zeros(2 * KERNEL_RADIUS + 1, dtype=torch.float64, device=device)
        for index in FIT_WINDOWS:
            entry = windows[index]
            shapes = [v2.dequantize_nvfp4(*entry[r]).shape for r in ("q", "k", "v")]
            pq = solution.hif4_dynamic_quantize_q(*entry["q"], q_heads, dim, states["q_state"])
            pk = solution.hif4_dynamic_quantize_k(*entry["k"], kv_heads, dim, states["k_state"])
            qd = dense(pq, shapes[0]).reshape(-1, q_heads, dim)
            kd = dense(pk, shapes[1]).reshape(-1, kv_heads, dim)
            tokens = int(shapes[0][0])
            keep = pick_tokens(tokens, FIT_TOKENS)
            idx = bucket_ids(tokens).to(device)
            for h in range(q_heads):
                p = probs_for(qd, kd, h, h // group)[keep]
                sub = idx[keep]
                for b in range(2 * KERNEL_RADIUS + 1):
                    sel = sub == b
                    if bool(sel.any()):
                        totals[h, b] += p[sel].sum()
                        if h == 0:
                            counts[b] += float(sel.sum())
        counts = counts.clamp_min(1.0)
        w = totals / counts[None, :]
        kernel_report[layer] = {
            "counts": [int(c) for c in counts.tolist()],
            "w_head0": [round(float(x), 6) for x in w[0].tolist()],
            "w_mean_over_heads": [round(float(x), 6) for x in w.mean(0).tolist()],
        }
        print(
            f"[L{layer:>2}] kernel fitted; head0 |r|<=3: "
            f"{[round(float(x), 4) for x in w[0, KERNEL_RADIUS - 3:KERNEL_RADIUS + 4].tolist()]}  "
            f"far={float(w[0, -1]):.4f}",
            flush=True,
        )

        # --- evaluate on held-out windows with the ACTUAL V residual
        for w_index in EVAL_WINDOWS:
            entry = windows[w_index]
            shapes = [v2.dequantize_nvfp4(*entry[r]).shape for r in ("q", "k", "v")]
            pq = solution.hif4_dynamic_quantize_q(*entry["q"], q_heads, dim, states["q_state"])
            pk = solution.hif4_dynamic_quantize_k(*entry["k"], kv_heads, dim, states["k_state"])
            pv = solution.hif4_dynamic_quantize_v(*entry["v"], kv_heads, dim, states["v_state"])
            q_ref = v2.dequantize_nvfp4(*entry["q"]).to(torch.float64).to(device)
            k_ref = v2.dequantize_nvfp4(*entry["k"]).to(torch.float64).to(device)
            v_ref = v2.dequantize_nvfp4(*entry["v"]).to(torch.float64).to(device)
            qd = dense(pq, shapes[0]).reshape(-1, q_heads, dim)
            kd = dense(pk, shapes[1]).reshape(-1, kv_heads, dim)
            v_hat = dense(pv, shapes[2]).reshape(-1, kv_heads, dim)
            dV = v_hat - v_ref.reshape(-1, kv_heads, dim)
            tokens = int(shapes[0][0])

            # TRUE, through the real forward (Q,K at deployed values, V varied)
            out_hat = attention(qd.reshape(tokens, -1), kd.reshape(tokens, -1), v_hat.reshape(tokens, -1))
            out_ref = attention(qd.reshape(tokens, -1), kd.reshape(tokens, -1), v_ref)
            true = float((out_hat - out_ref).square().sum())

            idx = bucket_ids(tokens).to(device)
            true_a = 0.0
            kernel = 0.0
            kernel_raw = 0.0
            uniform = 0.0
            surr_sq = 0.0
            for h in range(q_heads):
                g = h // group
                p = probs_for(qd, kd, h, g)
                dv = dV[:, g, :]
                true_a += float((p @ dv).square().sum())
                wk = w[h][idx]
                kernel_raw += float((wk @ dv).square().sum())
                # The least-squares fit minimises entrywise error and does NOT
                # force rows to sum to one, while the true A is row-stochastic:
                # on a long window the far bucket alone contributes
                # count * w_far ~ 1.8.  Rescaling each row to unit sum is the
                # same fit with that scale artefact removed.
                wk = wk / wk.sum(-1, keepdim=True).clamp_min(1e-30)
                kernel += float((wk @ dv).square().sum())
                # kernel fit error: how far the Toeplitz stand-in is from A on
                # THIS residual -- the quantity a rule driven by W would see.
                surr_sq += float(((wk - p) @ dv).square().sum())
                # Uniform kernel W[t,k] = 1/T gives the same (W dV)[t] on every
                # row, so the sum over rows is just T * ||mean_k dV[k]||^2.
                uniform += tokens * float(dv.mean(0).square().sum())
            rows.append(
                {
                    "layer": layer,
                    "window": w_index,
                    "tokens": tokens,
                    "true": true,
                    "true_a": true_a,
                    "kernel": kernel,
                    "kernel_raw": kernel_raw,
                    "uniform": uniform,
                    "ratio_true_a": true_a / true if true > 0 else float("nan"),
                    "ratio_kernel": kernel / true if true > 0 else float("nan"),
                    "ratio_kernel_raw": kernel_raw / true if true > 0 else float("nan"),
                    "ratio_uniform": uniform / true if true > 0 else float("nan"),
                    "surrogate_rel_err": math.sqrt(surr_sq / true_a) if true_a > 0 else float("nan"),
                }
            )
            print(
                f"  [L{layer:>2}/w{w_index} T={tokens:>4}] true={true:.6e}  "
                f"trueA/true={rows[-1]['ratio_true_a']:.4f}  "
                f"kernel/true={rows[-1]['ratio_kernel']:.4f} (raw {rows[-1]['ratio_kernel_raw']:.4f})  "
                f"uniform/true={rows[-1]['ratio_uniform']:.4f}  "
                f"surr_rel_err={rows[-1]['surrogate_rel_err']:.4f}",
                flush=True,
            )

    ta = [r["ratio_true_a"] for r in rows]
    ke = [r["ratio_kernel"] for r in rows]
    un = [r["ratio_uniform"] for r in rows]
    print()
    print("INDEPENDENT CHECK  true_a / true (must be ~1.000 if the A used is the scored attention):")
    print(f"  mean={statistics.mean(ta):.6f}  range=[{min(ta):.6f}, {max(ta):.6f}]  n={len(ta)}")
    print()
    print("predictor / TRUE, per window (1.000 = perfect):")
    print(f"  KERNEL   mean={statistics.mean(ke):.4f}  range=[{min(ke):.4f}, {max(ke):.4f}]")
    print(f"  UNIFORM  mean={statistics.mean(un):.4f}  range=[{min(un):.4f}, {max(un):.4f}]")
    se = [r["surrogate_rel_err"] for r in rows]
    print()
    print("kernel fit error  ||(W-A)dV|| / ||A dV||  (0 = the kernel reproduces A exactly):")
    print(f"  mean={statistics.mean(se):.4f}  range=[{min(se):.4f}, {max(se):.4f}]")
    print()
    print("per layer:")
    for layer in LAYERS:
        sub = [r for r in rows if r["layer"] == layer]
        if not sub:
            continue
        print(
            f"  L{layer:>2}  trueA/true={statistics.mean(x['ratio_true_a'] for x in sub):.4f}  "
            f"kernel/true={statistics.mean(x['ratio_kernel'] for x in sub):.4f}  "
            f"uniform/true={statistics.mean(x['ratio_uniform'] for x in sub):.4f}"
        )
    print()
    print("NO THRESHOLD IS APPLIED (plan section 6.1). These numbers describe the")
    print("fixed-kernel approximation on the sampled windows and nothing more.")

    (HERE / "vk1-diagnostic.json").write_text(
        json.dumps(
            {
                "layers": list(LAYERS),
                "fit_windows": list(FIT_WINDOWS),
                "eval_windows": list(EVAL_WINDOWS),
                "kernel_radius": KERNEL_RADIUS,
                "fit_tokens_cap": FIT_TOKENS,
                "kernel_blocks": kernel_report,
                "rows": rows,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"wrote {HERE / 'vk1-diagnostic.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
