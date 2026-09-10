"""Before opening V: how much of V's encoding error is the codec's, and how much is the calibration's?

The ceiling measurement showed that making V lossless would take the panel from
gain 0.5340 to 0.8115 -- V is where the Attention error lives.  That is an upper
bound on *any* V-side work, but it says nothing about how much of it is reachable
from calibration.  The question a card has to answer first is:

    is the current V encoding worse than the uncalibrated standard codec, or
    better?  And how far is either from what the format permits?

This script measures, per layer and averaged over the 72 attention cases:

  E_v_current   the V operand error of the deployed (calibrated) encoding
  E_v_standard  the V operand error of the plain encode_standard_hif4 codec
  E_v_scale     what the raw NVFP4 reference itself costs, for scale

Everything is operand-level MSE against the NVFP4 reference decode -- V carries no
learned rotation, so unlike Q/K this comparison is gauge-neutral and can be read
directly.

Reading it:

  * E_v_current >> E_v_standard -- calibration is actively hurting V, and there is
    obvious room.  That would be a defect, not a mechanism.
  * E_v_current ~  E_v_standard -- calibration is not buying anything on V; room,
    if any, is elsewhere in the format.
  * E_v_current << E_v_standard -- calibration already helps V; the remaining gap
    to lossless is what a better code assignment would have to close, and its size
    says whether it is plausible.

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


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def mse(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a - b).square().mean())


def main() -> int:
    torch.set_grad_enabled(False)
    solution = load_module(SOLUTION, "v_room_solution")

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
            row = {"layer": layer, "window": window}
            for index, (role, heads, api, state_name) in enumerate(
                (
                    ("q", q_heads, solution.hif4_dynamic_quantize_q, "q_state"),
                    ("k", kv_heads, solution.hif4_dynamic_quantize_k, "k_state"),
                    ("v", kv_heads, solution.hif4_dynamic_quantize_v, "v_state"),
                )
            ):
                pair = v2._pair(entry[index].to(torch.float32))
                reference = v2.dequantize_nvfp4(*pair).to(torch.float32)
                params = api(pair[0], pair[1], heads, head_dim, states[state_name])
                deployed = v2.dequantize_hif4(
                    v2._cpu_params(params), reference.shape
                ).to(torch.float32)
                standard = v2.decode_standard_hif4(
                    v2.encode_standard_hif4(reference)
                ).to(torch.float32)
                row[f"E_{role}_current"] = mse(deployed, reference)
                row[f"E_{role}_standard"] = mse(standard, reference)
            rows.append(row)

    keys = [f"E_{r}_{w}" for r in ("v", "q", "k") for w in ("current", "standard")]
    print(f"cases={len(rows)}   (operand-level MSE vs the NVFP4 reference decode)")
    print(f"{'layer':>5} " + " ".join(f"{k:>14}" for k in keys))
    for layer in layers:
        sub = [r for r in rows if r["layer"] == layer]
        if not sub:
            continue
        print(
            f"{layer:>5} "
            + " ".join(f"{statistics.mean(r[k] for r in sub):>14.6f}" for k in keys)
        )
    print(f"{'ALL':>5} " + " ".join(f"{statistics.mean(r[k] for r in rows):>14.6f}" for k in keys))

    overall = {k: statistics.mean(r[k] for r in rows) for k in keys}
    print()
    for role in ("v", "q", "k"):
        cur, std = overall[f"E_{role}_current"], overall[f"E_{role}_standard"]
        print(
            f"  {role}: current {cur:.6f} vs standard {std:.6f} -> "
            f"calibration is {'BETTER' if cur < std else 'WORSE'} by "
            f"{abs(cur - std) / std * 100:.1f}%"
        )

    (HERE / "v-room.json").write_text(
        json.dumps({"cases": len(rows), "overall": overall, "rows": rows}, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"\nwrote {HERE / 'v-room.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
