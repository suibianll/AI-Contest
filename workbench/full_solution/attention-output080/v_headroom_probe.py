"""Before writing a V card: is the 0.8115 ceiling nominal, or reachable inside the format?

The ceiling measurement says making V lossless would take the panel from 0.5340 to
0.8115.  That bounds any V-side work from above but says nothing about what is
reachable: if the HiF4 format itself cannot represent V much better than it does
now, the ceiling is nominal and a card would be wasted effort.

So probe the floor directly.  The V codec is `_nvfp4_to_hif4` with the V state's
knobs; two of them govern how much refinement it may spend (`max_refine_ratio`,
`max_refine_blocks`).  This script re-encodes V with those two widened by fixed
factors and re-scores the panel with everything else untouched -- same Q/K, same
reference, same standard, same metric.

This is a feasibility probe, not a card: a handful of fixed points to see whether
the curve moves at all, and it writes no candidate.  If V's error barely responds,
the honest conclusion is that the format is the limit and the 0.8115 ceiling
cannot be collected; if it responds strongly, there is room worth a card.

CPU only, no training, no shard.
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

SOLUTION = ROOT / "solution.py"
PACK = ROOT / "artifacts/official_eval/cache/qwen3.5-4b-proxy-v2.pt"

# (label, ratio multiplier, block multiplier)
VARIANTS = (
    ("current", 1.0, 1.0),
    ("ratio x2", 2.0, 1.0),
    ("ratio x5", 5.0, 1.0),
    ("blocks x4", 1.0, 4.0),
    ("ratio x5 + blocks x4", 5.0, 4.0),
)


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def main() -> int:
    torch.set_grad_enabled(False)
    solution = load_module(SOLUTION, "v_probe_solution")

    sys.path.insert(0, str(ROOT / "evaluator"))
    import official_eval as v2  # noqa: PLC0415

    pack = torch.load(PACK, map_location="cpu", mmap=True, weights_only=False)
    q_heads = int(pack["q_heads"])
    kv_heads = int(pack["kv_heads"])
    head_dim = int(pack["head_dim"])
    device = torch.device("cpu")
    layers = sorted({int(layer) for layer in pack["metadata"]["attention_layers"]})

    states_by_layer = {}
    for layer in layers:
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
        states_by_layer[layer] = solution.hif4_calibration_attention(
            windows, q_heads, kv_heads, head_dim
        )

    rows = []
    for layer in layers:
        for window in range(len(pack["test_qkv"])):
            entry = pack["test_qkv"][window][layer]
            if entry is None:
                continue
            states = states_by_layer[layer]
            pairs = [v2._pair(entry[i].to(torch.float32)) for i in range(3)]
            references = [v2.dequantize_nvfp4(*p).to(torch.float32) for p in pairs]
            standards = [
                v2.decode_standard_hif4(v2.encode_standard_hif4(r)).to(torch.float32)
                for r in references
            ]

            def attention(qkv):
                return v2._attention(*(t[None] for t in qkv), q_heads, kv_heads, head_dim)

            ref_out = attention(references)
            std_out = attention(standards)
            mse_std = float((std_out - ref_out).square().mean())

            q_hat = v2.dequantize_hif4(
                v2._cpu_params(
                    solution.hif4_dynamic_quantize_q(
                        pairs[0][0], pairs[0][1], q_heads, head_dim, states["q_state"]
                    )
                ),
                references[0].shape,
            ).to(torch.float32)
            k_hat = v2.dequantize_hif4(
                v2._cpu_params(
                    solution.hif4_dynamic_quantize_k(
                        pairs[1][0], pairs[1][1], kv_heads, head_dim, states["k_state"]
                    )
                ),
                references[1].shape,
            ).to(torch.float32)

            row = {"layer": layer, "window": window, "mse_standard": mse_std}
            base_state = states["v_state"]
            for label, ratio_mul, block_mul in VARIANTS:
                variant = dict(base_state)
                variant["max_refine_ratio"] = float(base_state["max_refine_ratio"]) * ratio_mul
                variant["max_refine_blocks"] = int(
                    int(base_state["max_refine_blocks"]) * block_mul
                )
                v_hat = v2.dequantize_hif4(
                    v2._cpu_params(
                        solution.hif4_dynamic_quantize_v(
                            pairs[2][0], pairs[2][1], kv_heads, head_dim, variant
                        )
                    ),
                    references[2].shape,
                ).to(torch.float32)
                play_out = attention([q_hat, k_hat, v_hat])
                mse_play = float((play_out - ref_out).square().mean())
                row[f"gain::{label}"] = 1.0 - mse_play / mse_std
                row[f"Ev::{label}"] = float(
                    (v_hat - references[2]).square().mean()
                )
            rows.append(row)

    print(f"cases={len(rows)}")
    labels = [v[0] for v in VARIANTS]
    print(f"{'layer':>5} " + " ".join(f"{l:>12}" for l in labels))
    for layer in layers:
        sub = [r for r in rows if r["layer"] == layer]
        if not sub:
            continue
        print(
            f"{layer:>5} "
            + " ".join(
                f"{statistics.mean(r[f'gain::{l}'] for r in sub):>12.4f}" for l in labels
            )
        )
    print(
        f"{'ALL':>5} "
        + " ".join(
            f"{statistics.mean(r[f'gain::{l}'] for r in rows):>12.4f}" for l in labels
        )
    )
    print("\nV operand error (E_v) and its response:")
    print(f"{'layer':>5} " + " ".join(f"{l:>12}" for l in labels))
    for layer in layers:
        sub = [r for r in rows if r["layer"] == layer]
        if not sub:
            continue
        print(
            f"{layer:>5} "
            + " ".join(
                f"{statistics.mean(r[f'Ev::{l}'] for r in sub):>12.6f}" for l in labels
            )
        )
    print(
        f"{'ALL':>5} "
        + " ".join(
            f"{statistics.mean(r[f'Ev::{l}'] for r in rows):>12.6f}" for l in labels
        )
    )
    print("\nreference points: current gain 0.5340 | V lossless 0.8115 | target 0.80")

    payload = {
        "variants": labels,
        "cases": len(rows),
        "overall_gain": {
            l: statistics.mean(r[f"gain::{l}"] for r in rows) for l in labels
        },
        "overall_Ev": {l: statistics.mean(r[f"Ev::{l}"] for r in rows) for l in labels},
        "rows": rows,
    }
    (HERE / "v-headroom.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"\nwrote {HERE / 'v-headroom.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
