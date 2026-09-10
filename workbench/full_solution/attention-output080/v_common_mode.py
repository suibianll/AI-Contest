"""The fastest decisive number: how much of V's output error is common-mode?

The objective is the attention output MSE.  With the parent probabilities A fixed,

    O = A V ,  delta_O = A delta_V ,  ||delta_O||^2 = ||A delta_V||^2

and A's rows sum to one, so delta_V splits into two parts with opposite fates:

  * the part that is constant across key tokens -- passes through at full weight;
  * the part that oscillates across key tokens -- partially cancels.

Nearest-element rounding (what operand-MSE driving produces) makes each element's
error independent, so the common-mode part grows like sqrt(n) while the oscillating
part cancels.  The V encoder sees the whole matrix in one call, so it could instead
choose codes whose per-channel error is near zero-mean across tokens -- removing the
common-mode error without touching the operand error much.

This measures, on real cases, the two output errors:

  E_current   ||A (V_hat - V_ref)||^2                what we ship
  E_balanced  ||A (delta - colmean(delta))||^2       if the per-channel mean were removed

and the panel MSE with the balanced V actually constructed and run through the same
attention.  If E_balanced << E_current, common-mode removal is a real, deployable
lever -- and one that operand-optimal rounding structurally cannot reach.

CPU only, no training, no shard, no candidate.
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
    solution = load("cm_solution", ROOT / "solution.py")

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
            refs = [v2.dequantize_nvfp4(*p).to(torch.float32) for p in pairs]

            pq = solution.hif4_dynamic_quantize_q(*pairs[0], q_heads, head_dim, states["q_state"])
            pk = solution.hif4_dynamic_quantize_k(*pairs[1], kv_heads, head_dim, states["k_state"])
            pv = solution.hif4_dynamic_quantize_v(*pairs[2], kv_heads, head_dim, states["v_state"])
            q_hat = v2.dequantize_hif4(v2._cpu_params(pq), refs[0].shape).to(torch.float32)
            k_hat = v2.dequantize_hif4(v2._cpu_params(pk), refs[1].shape).to(torch.float32)
            v_hat = v2.dequantize_hif4(v2._cpu_params(pv), refs[2].shape).to(torch.float32)
            v_ref = refs[2]

            standards = [
                v2.decode_standard_hif4(v2.encode_standard_hif4(r)).to(torch.float32)
                for r in refs
            ]
            target = attention(refs[0], refs[1], refs[2])
            std_out = attention(standards[0], standards[1], standards[2])
            mse_std = float((std_out - target).square().mean())

            delta = v_hat - v_ref                      # (tokens_k, channels)
            col_mean = delta.mean(dim=0, keepdim=True)
            v_bal = v_hat - col_mean                   # same codes' values, mean removed

            mse_cur = float((attention(q_hat, k_hat, v_hat) - target).square().mean())
            mse_bal = float((attention(q_hat, k_hat, v_bal) - target).square().mean())

            # V-alone output error, current vs mean-removed.
            e_cur = float((v_hat - v_ref).square().sum().item())        # operand MSE * n
            rows.append(
                {
                    "layer": layer,
                    "window": window,
                    "gain_current": 1.0 - mse_cur / mse_std,
                    "gain_balanced": 1.0 - mse_bal / mse_std,
                    "operand_mse_current": float(delta.square().mean()),
                    "operand_mse_balanced": float((delta - col_mean).square().mean()),
                    "common_mode_share": float(col_mean.square().mean() / delta.square().mean().clamp_min(1e-30)),
                }
            )

    cur = statistics.mean(r["gain_current"] for r in rows)
    bal = statistics.mean(r["gain_balanced"] for r in rows)
    print(f"cases={len(rows)}")
    print(f"gain current  = {cur:.4f}")
    print(f"gain balanced = {bal:.4f}   ({bal - cur:+.4f}, target 0.80)")
    print()
    print(f"operand MSE current  = {statistics.mean(r['operand_mse_current'] for r in rows):.8f}")
    print(f"operand MSE balanced = {statistics.mean(r['operand_mse_balanced'] for r in rows):.8f}")
    print(f"common-mode share of V's error = {statistics.mean(r['common_mode_share'] for r in rows):.4f}")
    print()
    per_layer = {}
    for r in rows:
        per_layer.setdefault(r["layer"], []).append(r)
    for layer, sub in sorted(per_layer.items()):
        print(
            f"  layer {layer:>2}: gain {statistics.mean(x['gain_current'] for x in sub):.4f} "
            f"-> {statistics.mean(x['gain_balanced'] for x in sub):.4f}  "
            f"(common-mode share {statistics.mean(x['common_mode_share'] for x in sub):.3f})"
        )
    print()
    print(
        "note: `balanced` here removes the mean by *shifting values*, which is not a legal\n"
        "      code change -- it measures the headroom of the mechanism, not the mechanism.\n"
        "      Realising it needs codes whose per-channel error is mean-zero."
    )

    (HERE / "v-common-mode.json").write_text(
        json.dumps({"cases": len(rows), "gain_current": cur, "gain_balanced": bal, "rows": rows}, indent=2)
        + "\n",
        encoding="utf-8",
    )
    print(f"wrote {HERE / 'v-common-mode.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
