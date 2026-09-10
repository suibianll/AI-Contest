"""Is V's remaining error in the scales, and if so how much?

The element-level assignment was proved optimal *at the encoder's own scales*
(see v-conclusion.md section 3ter).  That leaves the scales themselves -- the
E6M2 `scale_factor` and the two levels of `lv2` / `lv3` -- and neither of them has
been measured.

This probe holds the element-level choice at its optimum and asks a narrow
question: if the block's `scale_factor` were multiplied by a fixed set of factors
(the lv2/lv3 hierarchy left as the encoder chose it, so the search is over the
block scale only), how much would the block's error fall?

  * if the best factor is 1x on most blocks, the encoder's block scale is already
    the right one and the residual really is the representation's;
  * if a shifted factor wins consistently, the block-scale rule is leaving room,
    and choosing it by the block error instead of by the amax rule is a real,
    already-authorised lever.

Nothing is trained, nothing is written as a candidate, no shard is run.
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

LAYERS = (22, 15, 0)
FACTORS = (0.25, 0.3535533905932738, 0.5, 0.7071067811865476, 1.0,
           1.4142135623730951, 2.0, 2.8284271247461903, 4.0)
LEVELS = torch.arange(8, dtype=torch.float32) * 0.25


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def best_error_at(scale, target):
    """Best achievable squared error at a fixed per-element scale (float64)."""

    cand = LEVELS.reshape(1, 1, 1, 1, 1, 8) * scale.unsqueeze(-1)
    idx = (cand - target.abs().unsqueeze(-1)).abs().argmin(-1, keepdim=True)
    best = cand.gather(-1, idx).squeeze(-1) * torch.sign(target)
    return (best.double() - target.double()).square()


def main() -> int:
    torch.set_grad_enabled(False)
    solution = load_module(SOLUTION, "scale_solution")

    sys.path.insert(0, str(ROOT / "evaluator"))
    import official_eval as v2  # noqa: PLC0415

    pack = torch.load(PACK, map_location="cpu", mmap=True, weights_only=False)
    q_heads = int(pack["q_heads"])
    kv_heads = int(pack["kv_heads"])
    head_dim = int(pack["head_dim"])
    device = torch.device("cpu")

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
            base = (
                params["scale_lv3"].to(torch.float32)
                * params["scale_lv2"].to(torch.float32)
                * params["scale_factor"].to(torch.float32)
            )
            mant = params["mant"].to(torch.float32)
            # Element shape is (..., 8, 2, 4); the scale block stops one axis short.
            target = reference.reshape(*base.shape[:-1], 4)
            scale_full = base.expand_as(target)

            current = (
                params["sign"].to(torch.float32) * mant * scale_full
            )
            e_current = float(
                (current.double() - target.double()).square().mean()
            )

            per_factor = {}
            for factor in FACTORS:
                scaled = scale_full * factor
                err = best_error_at(scaled, target)
                per_factor[factor] = float(err.mean())
                # block-level view: which factor wins per block
            best_factor = min(per_factor, key=per_factor.get)
            rows.append(
                {
                    "layer": layer,
                    "window": window,
                    "E_current": e_current,
                    "E_best": per_factor[best_factor],
                    "best_factor": best_factor,
                    "per_factor": {str(k): v for k, v in per_factor.items()},
                }
            )

    print(f"cases={len(rows)}")
    cur = statistics.mean(r["E_current"] for r in rows)
    best = statistics.mean(r["E_best"] for r in rows)
    print(f"E_current (float64) = {cur:.10f}")
    print(f"E_best over block-scale factors = {best:.10f}")
    print(f"headroom = {100 * (cur - best) / cur:.4f}%")
    print()
    counts = {}
    for r in rows:
        counts[r["best_factor"]] = counts.get(r["best_factor"], 0) + 1
    print("winning block-scale factor (per case):")
    for factor in sorted(counts):
        print(f"  x{factor:<8.4f}: {counts[factor]:>3} cases")

    per_layer = {}
    for r in rows:
        per_layer.setdefault(r["layer"], []).append(r)
    print()
    for layer, sub in sorted(per_layer.items()):
        c = statistics.mean(x["E_current"] for x in sub)
        b = statistics.mean(x["E_best"] for x in sub)
        print(f"  layer {layer:>2}: E_current {c:.10f} -> best {b:.10f}  ({100*(c-b)/c:.4f}% lower)")

    (HERE / "v-scale-headroom.json").write_text(
        json.dumps({"cases": len(rows), "E_current": cur, "E_best": best, "rows": rows}, indent=2)
        + "\n",
        encoding="utf-8",
    )
    print(f"\nwrote {HERE / 'v-scale-headroom.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
