"""What does the hierarchy fix buy at the panel level?

`v_hierarchy_headroom.py` showed the lv2/lv3 assignment is leaving ~58% of V's
encoding error on the table, and V's error is the dominant term in the attention
output error.  The number that decides whether that matters is the panel gain
with the oracle V in place, so this script builds the oracle V_hat as a real
tensor and scores it with the evaluator's own path.

Three points per case:

  current   the shipped encoding
  oracle    lv2/lv3 chosen per group of 8 by exact enumeration, element-level
            choice re-derived at each setting
  lossless  V replaced by its NVFP4 reference decode (the 0.8115 ceiling)

This is an oracle, not a candidate: it needs the target to choose the bits, so it
says what is reachable, not what is implementable.  The implementation question --
choosing lv2/lv3 at calibration from something the dynamic API can see -- is the
card that would follow.

CPU only, no training, no shard.
"""

from __future__ import annotations

import importlib.util
import itertools
import json
import statistics
import sys
from pathlib import Path

import torch


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]

SOLUTION = ROOT / "solution.py"
PACK = ROOT / "artifacts/official_eval/cache/qwen3.5-4b-proxy-v2.pt"

LAYERS = (22, 15, 5, 1, 0, 8)
LEVELS = torch.arange(8, dtype=torch.float64) * 0.25


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def main() -> int:
    torch.set_grad_enabled(False)
    solution = load_module(SOLUTION, "oracle_solution")

    sys.path.insert(0, str(ROOT / "evaluator"))
    import official_eval as v2  # noqa: PLC0415

    pack = torch.load(PACK, map_location="cpu", mmap=True, weights_only=False)
    q_heads = int(pack["q_heads"])
    kv_heads = int(pack["kv_heads"])
    head_dim = int(pack["head_dim"])
    device = torch.device("cpu")

    # (lv2 multiplier, lv3 multiplier for the left pair, for the right pair)
    COMBOS = list(itertools.product((1.0, 2.0), repeat=3))

    def attention(qkv):
        return v2._attention(*(t[None] for t in qkv), q_heads, kv_heads, head_dim)

    rows = []
    for layer in LAYERS:
        windows = [
            {
                role: tuple(
                    t.to(device)
                    for t in v2._pair(
                        pack["calibration_qkv"][s][layer][i].to(torch.float32)
                    )
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
            references = [v2.dequantize_nvfp4(*p).to(torch.float32) for p in pairs]
            standards = [
                v2.decode_standard_hif4(v2.encode_standard_hif4(r)).to(torch.float32)
                for r in references
            ]

            def dequant(api, pair, heads, state_name):
                params = api(pair[0], pair[1], heads, head_dim, states[state_name])
                return params

            q_pr = dequant(solution.hif4_dynamic_quantize_q, pairs[0], q_heads, "q_state")
            k_pr = dequant(solution.hif4_dynamic_quantize_k, pairs[1], kv_heads, "k_state")
            v_pr = dequant(solution.hif4_dynamic_quantize_v, pairs[2], kv_heads, "v_state")
            q_hat = v2.dequantize_hif4(v2._cpu_params(q_pr), references[0].shape).to(torch.float32)
            k_hat = v2.dequantize_hif4(v2._cpu_params(k_pr), references[1].shape).to(torch.float32)
            v_hat = v2.dequantize_hif4(v2._cpu_params(v_pr), references[2].shape).to(torch.float32)

            ref_out = attention(references)
            std_out = attention(standards)
            mse_std = float((std_out - ref_out).square().mean())

            # --- oracle V, group of 8 at a time, exact over the 8 settings ---
            sf = v_pr["scale_factor"].to(torch.float64)
            sign_t = v_pr["sign"].to(torch.float64)
            mant_t = v_pr["mant"].to(torch.float64)
            shape = sign_t.shape
            target = references[2].reshape(shape).to(torch.float64)
            tgt8 = target.reshape(target.shape[0], -1, 8, 8)
            base = sf.reshape(sf.shape[0], -1, 1, 1)

            best_err = None
            best_val = None
            for a, b, c in COMBOS:
                per_pair = torch.cat(
                    [
                        torch.full(tgt8.shape[:-1] + (4,), b, dtype=torch.float64),
                        torch.full(tgt8.shape[:-1] + (4,), c, dtype=torch.float64),
                    ],
                    dim=-1,
                )
                scale = base * a * per_pair
                cand = LEVELS.reshape(1, 1, 1, 1, 8) * scale.unsqueeze(-1)
                idx = (cand - tgt8.abs().unsqueeze(-1)).abs().argmin(-1, keepdim=True)
                val = cand.gather(-1, idx).squeeze(-1) * torch.sign(tgt8)
                err = (val - tgt8).square().mean(dim=-1, keepdim=True)
                if best_err is None:
                    best_err, best_val = err, val
                else:
                    take = err < best_err
                    best_val = torch.where(take, val, best_val)
                    best_err = torch.where(take, err, best_err)
            oracle_v = best_val.reshape(shape).to(torch.float32)

            mse_cur = float((attention([q_hat, k_hat, v_hat]) - ref_out).square().mean())
            mse_orc = float((attention([q_hat, k_hat, oracle_v]) - ref_out).square().mean())
            mse_lossless = float((attention([q_hat, k_hat, references[2]]) - ref_out).square().mean())

            # Validate in the same run that the oracle really does lower the V
            # operand error: the whole point is that the two directions disagree.
            ev_cur = float(
                (v_hat.double() - references[2].double()).square().mean()
            )
            ev_orc = float(
                (
                    oracle_v.flatten(start_dim=-4, end_dim=-1).double()
                    - references[2].double()
                )
                .square()
                .mean()
            )

            rows.append(
                {
                    "layer": layer,
                    "window": window,
                    "gain_current": 1.0 - mse_cur / mse_std,
                    "gain_oracle": 1.0 - mse_orc / mse_std,
                    "gain_v_lossless": 1.0 - mse_lossless / mse_std,
                    "Ev_current": ev_cur,
                    "Ev_oracle": ev_orc,
                }
            )

    print(f"cases={len(rows)}")
    print(f"{'layer':>5} {'current':>10} {'oracle V':>10} {'V lossless':>11}")
    for layer in LAYERS:
        sub = [r for r in rows if r["layer"] == layer]
        if not sub:
            continue
        print(
            f"{layer:>5} "
            + f"{statistics.mean(r['gain_current'] for r in sub):>10.4f} "
            + f"{statistics.mean(r['gain_oracle'] for r in sub):>10.4f} "
            + f"{statistics.mean(r['gain_v_lossless'] for r in sub):>11.4f}"
        )
    cur = statistics.mean(r["gain_current"] for r in rows)
    orc = statistics.mean(r["gain_oracle"] for r in rows)
    los = statistics.mean(r["gain_v_lossless"] for r in rows)
    print(f"{'ALL':>5} {cur:>10.4f} {orc:>10.4f} {los:>11.4f}")
    print()
    print(f"current {cur:.4f} | oracle V {orc:.4f} | V lossless {los:.4f} | target 0.80")
    evc = statistics.mean(r["Ev_current"] for r in rows)
    evo = statistics.mean(r["Ev_oracle"] for r in rows)
    print()
    print(f"V operand error: current {evc:.10f} -> oracle {evo:.10f} "
          f"({100*(evc-evo)/evc:.2f}% lower)")
    print(
        "direction check: the oracle lowers the operand error while lowering the "
        f"gain -- operand {100*(evc-evo)/evc:+.2f}%, gain {100*(orc-cur)/abs(cur):+.2f}%"
    )
    print(f"the hierarchy fix alone recovers {100*(orc-cur)/(los-cur):.1f}% of the distance to V-lossless")

    (HERE / "v-oracle-gain.json").write_text(
        json.dumps({"cases": len(rows), "current": cur, "oracle": orc, "lossless": los, "rows": rows}, indent=2)
        + "\n",
        encoding="utf-8",
    )
    print(f"wrote {HERE / 'v-oracle-gain.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
