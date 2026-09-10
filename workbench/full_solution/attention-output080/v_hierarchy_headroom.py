"""INVALID ORACLE -- retained only as the record of a mistake.  See v-conclusion.md section 3quater.

The `min` over the eight settings is taken PER ELEMENT here, which lets every element pick its
own lv2 and lv3.  The format does not allow that: lv2 is one bit per group of 8 and lv3 one bit
per pair of 4, so the choice must be shared inside the group.  The 58% headroom this script
reported is therefore an artefact of an illegal oracle and is retracted.  The valid version --
min per group -- is v_oracle_gain.py, which finds ~0.00%.


Two things are already proved: the element-level (sign, mantissa) choice is
optimal at the encoder's scales, and the block-level `scale_factor` is optimal up
to a uniform factor (x1.0 wins everywhere, because an amax-set block scale is
saturated in both directions).

What has not been measured is the hierarchy *between* them.  Within a 64-element
block the format offers two binary sub-levels:

    lv2   one bit per group of 8 elements
    lv3   one bit per pair of 4 elements

so each group of 8 has 2 * 2 * 2 = 8 possible (lv2, lv3_left, lv3_right)
settings.  Eight is small enough to enumerate exactly, per group, with the
element-level choice re-derived optimally at each setting.  That makes this an
exact oracle for the hierarchy, not a heuristic one.

If the oracle matches the encoder, the whole V encoding is optimal at every level
and the residual is the representation's -- the strongest form of the V
conclusion.  If it beats the encoder, the hierarchy rule is a real,
already-authorised lever.

CPU only, no training, no candidate, no shard.
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

LAYERS = (22, 15, 0)
LEVELS = torch.arange(8, dtype=torch.float64) * 0.25


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def best_error(scale, target):
    """Squared error of the best element-level choice, at fixed per-element scale."""

    cand = LEVELS.reshape(1, 8) * scale.reshape(-1, 1)
    tgt = target.reshape(-1, 1)
    idx = (cand - tgt.abs()).abs().argmin(-1, keepdim=True)
    best = cand.gather(-1, idx).squeeze(-1) * torch.sign(target.reshape(-1))
    return (best - target.reshape(-1)).square()


def main() -> int:
    raise SystemExit(
        "This script's oracle is illegal (the min is taken per element, so every "
        "element picks its own lv2/lv3; the format shares those inside a group of 8). "
        "Its 58% headroom reading is retracted -- see v-conclusion.md section 3quater. "
        "Use v_oracle_gain.py, whose oracle takes the min per group."
    )

    torch.set_grad_enabled(False)
    solution = load_module(SOLUTION, "hier_solution")

    sys.path.insert(0, str(ROOT / "evaluator"))
    import official_eval as v2  # noqa: PLC0415

    pack = torch.load(PACK, map_location="cpu", mmap=True, weights_only=False)
    q_heads = int(pack["q_heads"])
    kv_heads = int(pack["kv_heads"])
    head_dim = int(pack["head_dim"])
    device = torch.device("cpu")

    COMBOS = list(itertools.product((1.0, 2.0), repeat=3))  # (lv2, lv3_left, lv3_right)

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
            pair = v2._pair(entry[2].to(torch.float32))
            reference = v2.dequantize_nvfp4(*pair).to(torch.float32)
            params = solution.hif4_dynamic_quantize_v(
                pair[0], pair[1], kv_heads, head_dim, states["v_state"]
            )

            sf = params["scale_factor"].to(torch.float64)          # (rows, blocks,1,1,1)
            lv2 = params["scale_lv2"].to(torch.float64)            # (rows, blocks,8,1,1)
            lv3 = params["scale_lv3"].to(torch.float64)            # (rows, blocks,8,2,1)
            sign = params["sign"].to(torch.float64)
            mant = params["mant"].to(torch.float64)

            # (rows, blocks, 16 groups-of-8 ... reshape to 8 groups x 2 pairs x 4)
            target = reference.reshape(sign.shape).to(torch.float64)
            current = sign * mant * lv3 * lv2 * sf
            e_current = float((current - target).square().mean())

            # Oracle: per block, per group of 8, enumerate the 8 settings.
            tgt = target.reshape(target.shape[0], -1, 8, 8)        # (rows, blocks, 8 groups, 8 elems)
            base = sf.reshape(sf.shape[0], -1, 1, 1)               # block scale
            oracle_err = 0.0
            total = 0
            for ai, (a, b, c) in enumerate(COMBOS):
                # element scale = sf * lv2 * lv3, where lv3 differs per pair of 4
                per_pair = torch.cat(
                    [
                        torch.full((tgt.shape[0], tgt.shape[1], tgt.shape[2], 4), b, dtype=torch.float64),
                        torch.full((tgt.shape[0], tgt.shape[1], tgt.shape[2], 4), c, dtype=torch.float64),
                    ],
                    dim=-1,
                )
                scale = base * a * per_pair
                err = best_error(scale, tgt).reshape(tgt.shape[0], tgt.shape[1], tgt.shape[2], 8)
                if ai == 0:
                    best = err
                else:
                    best = torch.minimum(best, err)
            oracle_err = float(best.mean())
            total = best.numel()

            rows.append(
                {
                    "layer": layer,
                    "window": window,
                    "E_current": e_current,
                    "E_oracle": oracle_err,
                    "elements": total,
                }
            )

    cur = statistics.mean(r["E_current"] for r in rows)
    orc = statistics.mean(r["E_oracle"] for r in rows)
    print(f"cases={len(rows)}")
    print(f"E_current (float64) = {cur:.10f}")
    print(f"E_oracle  (float64) = {orc:.10f}")
    print(f"hierarchy headroom  = {100 * (cur - orc) / cur:.6f}%")
    print()
    per_layer = {}
    for r in rows:
        per_layer.setdefault(r["layer"], []).append(r)
    for layer, sub in sorted(per_layer.items()):
        c = statistics.mean(x["E_current"] for x in sub)
        o = statistics.mean(x["E_oracle"] for x in sub)
        print(f"  layer {layer:>2}: {c:.10f} -> {o:.10f}  ({100*(c-o)/c:.6f}% lower)")

    (HERE / "v-hierarchy-headroom.json").write_text(
        json.dumps({"cases": len(rows), "E_current": cur, "E_oracle": orc, "rows": rows}, indent=2)
        + "\n",
        encoding="utf-8",
    )
    print(f"\nwrote {HERE / 'v-hierarchy-headroom.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
