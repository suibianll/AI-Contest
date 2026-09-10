"""Is V's remaining error the format's floor, or the encoder's rounding choice?

The V side is the only path to 0.80, and the question that decides whether any
encoder-side work can help is: **given the scales the encoder already chose, is
each element's (sign, mantissa) the best one available in the format?**

Each decoded element is `sign * mant * scale_lv3 * scale_lv2 * scale_factor`, with
`mant` in {0, 0.25, ..., 1.75} (eight levels) and `sign` in {-1, 0, 1}.  So for a
fixed per-element scale `S` the best possible reconstruction of a target value is
the nearest of those 25 points -- computable exactly, element by element.

Two numbers come out per case:

  E_current   what the encoder actually produced
  E_oracle    the best any (sign, mantissa) choice could do at the same scales

If they are close, the element-level assignment is already optimal and the残り
error is the *scales'* -- i.e. the format's, not the encoder's.  If E_oracle is
much smaller, the encoder is leaving real room on the table and an encoder-side
card has something to work with.

This holds the scales fixed on purpose: changing them is a different (larger)
question, and this measurement is the cheap one that decides whether to ask it.

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

LAYERS = (22, 15, 5, 1, 0, 8)
LEVELS = torch.arange(8, dtype=torch.float32) * 0.25


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def main() -> int:
    torch.set_grad_enabled(False)
    solution = load_module(SOLUTION, "floor_solution")

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

            sign = params["sign"].to(torch.float32)
            mant = params["mant"].to(torch.float32)
            scale = (
                params["scale_lv3"].to(torch.float32)
                * params["scale_lv2"].to(torch.float32)
                * params["scale_factor"].to(torch.float32)
            )
            # `_dequantize_hif4` flattens the last four dims; mirror it so the
            # element order matches the reference's (rows, channels).
            current = (sign * mant * scale).flatten(start_dim=-4, end_dim=-1)
            target = reference.reshape(current.shape)

            # The per-element scale, broadcast to the mantissa's shape, so each
            # element is compared against its own scale.
            s = scale.expand_as(mant).flatten(start_dim=-4, end_dim=-1)
            mag = (LEVELS.reshape(8, 1, 1, 1, 1, 1) * s.unsqueeze(0))
            # best magnitude per element (drop the level axis), then apply the
            # target's sign so a negative target is reconstructed negatively.
            diff = (mag - target.abs().unsqueeze(0)).abs()
            best_mag = mag.gather(0, diff.argmin(0, keepdim=True)).squeeze(0)
            oracle = best_mag * torch.sign(target)

            rows.append(
                {
                    "layer": layer,
                    "window": window,
                    "E_current": float((current - target).square().mean()),
                    "E_oracle": float((oracle - target).square().mean()),
                }
            )

    print(f"cases={len(rows)}")
    print(f"{'layer':>5} {'E_current':>14} {'E_oracle':>14} {'floor share':>12}")
    for layer in LAYERS:
        sub = [r for r in rows if r["layer"] == layer]
        if not sub:
            continue
        cur = statistics.mean(r["E_current"] for r in sub)
        ora = statistics.mean(r["E_oracle"] for r in sub)
        print(f"{layer:>5} {cur:>14.8f} {ora:>14.8f} {ora / cur:>11.1%}")
    cur = statistics.mean(r["E_current"] for r in rows)
    ora = statistics.mean(r["E_oracle"] for r in rows)
    print(f"{'ALL':>5} {cur:>14.8f} {ora:>14.8f} {ora / cur:>11.1%}")
    print()
    print(
        f"encoder is within {(1 - ora / cur) * 100:.1f}% of the best possible (sign, mantissa) "
        f"at its own scales"
    )

    (HERE / "v-format-floor.json").write_text(
        json.dumps({"cases": len(rows), "E_current": cur, "E_oracle": ora, "rows": rows}, indent=2)
        + "\n",
        encoding="utf-8",
    )
    print(f"wrote {HERE / 'v-format-floor.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
