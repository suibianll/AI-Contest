"""Split the +0.0038 into mechanism, proxy and search -- measure all three.

`v_error_balance.py` reported +0.0038 against a +0.0412 ceiling and I attributed
the gap to "code-step granularity".  That was over-attribution: three different
things could produce the same shortfall and I never separated them.

On the same cases, this measures three quantities before and after the balancing
pass, so the gap lands on one of them:

  1. TRUE   ||A delta_c||^2 summed over channels -- the actual quantity the
            attention output error is made of.  A is the parent's own softmax
            probabilities with the causal mask, per KV head (GQA: the group of Q
            heads that share it).
  2. PROXY  (sum_k delta_c)^2 per channel -- the unweighted surrogate the encoder
            can actually compute, which is exactly (1/k)-weighted attention.
  3. SEARCH how far the greedy moved the proxy.

Reading:
  * proxy well reduced, true not   -> the proxy is the bottleneck: closing it needs
                                      A, i.e. another API's input.  Mechanism not
                                      deployable in this form.
  * proxy barely reduced           -> the search is too weak; a better solver helps.
  * both reduced, gain unmoved     -> the gain or the ceiling measurement is wrong.

CPU only, no shard, no candidate.
"""

from __future__ import annotations

import importlib.util
import json
import math
import statistics
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]

MAX_TOKENS = 256
MAX_ITER = 6


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


balance = load("pb_balance", HERE / "v_error_balance.py")


def true_error(probs, delta, kv_heads, head_dim):
    """||A delta||^2 summed over KV heads and channels, float64."""

    k, _ = delta.shape
    group = probs.shape[0] // kv_heads
    a6 = probs.reshape(kv_heads, group, probs.shape[1], k).double()
    d3 = delta.reshape(k, kv_heads, head_dim).double().permute(1, 0, 2)  # (kv, k, d)
    total = 0.0
    for head in range(kv_heads):
        prod = torch.einsum("gqk,kd->gqd", a6[head], d3[head])
        total += float(prod.square().sum())
    return total


def proxy_error(delta):
    """Sum of (sum_k delta)^2 over channels -- the A-free surrogate."""

    return float(delta.double().sum(dim=0).square().sum())


def main() -> int:
    torch.set_num_threads(2)
    torch.set_grad_enabled(False)
    solution = load("pd_solution", ROOT / "solution.py")

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

    def probabilities(q_hat, k_hat):
        q = q_hat.reshape(-1, q_heads, head_dim).transpose(0, 1)
        k = k_hat.reshape(-1, kv_heads, head_dim).transpose(0, 1)
        k = k.repeat_interleave(q_heads // kv_heads, dim=0)
        logits = q @ k.transpose(-1, -2) / math.sqrt(head_dim)
        mask = torch.ones(logits.shape[-2], logits.shape[-1], dtype=torch.bool).triu(1)
        return logits.masked_fill(mask, -torch.inf).softmax(-1)

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
            pq = solution.hif4_dynamic_quantize_q(*pairs[0], q_heads, head_dim, states["q_state"])
            pk = solution.hif4_dynamic_quantize_k(*pairs[1], kv_heads, head_dim, states["k_state"])
            pv = solution.hif4_dynamic_quantize_v(*pairs[2], kv_heads, head_dim, states["v_state"])
            q_hat = v2.dequantize_hif4(v2._cpu_params(pq), refs[0].shape).to(torch.float32)
            k_hat = v2.dequantize_hif4(v2._cpu_params(pk), refs[1].shape).to(torch.float32)
            v_hat = v2.dequantize_hif4(v2._cpu_params(pv), refs[2].shape).to(torch.float32)

            probs = probabilities(q_hat, k_hat)
            d_cur = v_hat - refs[2]

            bal, flips = balance.balanced_params(pv, refs[2])
            v_bal = v2.dequantize_hif4(v2._cpu_params(bal), refs[2].shape).to(torch.float32)
            d_bal = v_bal - refs[2]

            rows.append(
                {
                    "layer": layer,
                    "window": window,
                    "flips": flips,
                    "true_current": true_error(probs, d_cur, kv_heads, head_dim),
                    "true_balanced": true_error(probs, d_bal, kv_heads, head_dim),
                    "proxy_current": proxy_error(d_cur),
                    "proxy_balanced": proxy_error(d_bal),
                    "operand_current": float(d_cur.double().square().mean()),
                    "operand_balanced": float(d_bal.double().square().mean()),
                }
            )

    def agg(key):
        return statistics.mean(r[key] for r in rows)

    print(f"cases={len(rows)}")
    print(f"TRUE  ||A d||^2   {agg('true_current'):.6f} -> {agg('true_balanced'):.6f}  "
          f"({100*(agg('true_balanced')-agg('true_current'))/agg('true_current'):+.2f}%)")
    print(f"PROXY (sum d)^2   {agg('proxy_current'):.6f} -> {agg('proxy_balanced'):.6f}  "
          f"({100*(agg('proxy_balanced')-agg('proxy_current'))/agg('proxy_current'):+.2f}%)")
    print(f"OPERAND mse       {agg('operand_current'):.8f} -> {agg('operand_balanced'):.8f}  "
          f"({100*(agg('operand_balanced')-agg('operand_current'))/agg('operand_current'):+.2f}%)")
    print(f"flips/case        {agg('flips'):.0f}")
    print()
    better_true = sum(1 for r in rows if r["true_balanced"] < r["true_current"])
    better_proxy = sum(1 for r in rows if r["proxy_balanced"] < r["proxy_current"])
    print(f"cases where TRUE improved:  {better_true}/{len(rows)}")
    print(f"cases where PROXY improved: {better_proxy}/{len(rows)}")
    print()
    print("reading: proxy well reduced but true not -> the proxy is the bottleneck (needs A, cross-API);")
    print("         proxy barely reduced              -> the search is weak;")
    print("         both reduced, gain unmoved        -> the gain or ceiling measurement is wrong.")

    (HERE / "v-proxy-diagnosis.json").write_text(
        json.dumps({"cases": len(rows), "rows": rows}, indent=2) + "\n", encoding="utf-8"
    )
    print(f"\nwrote {HERE / 'v-proxy-diagnosis.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
