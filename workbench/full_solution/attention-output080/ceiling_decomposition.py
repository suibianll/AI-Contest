"""Where does the Attention output error actually live, and what is each operand's ceiling?

The panel metric is `gain = 1 - mse_player / mse_standard`, where `standard` is the
*uncalibrated* standard HiF4 codec applied to Q/K/V and `player` is our calibrated
one.  So `gain = 0.80` means the calibrated encoding's attention error is 0.20x the
uncalibrated codec's.

Before building any mechanism that acts on an operand, the number that decides
whether it can matter at all is the ceiling: **if that operand were encoded
perfectly (replaced by its NVFP4 reference decode) and everything else stayed as
it is, what gain would result?**  An operand whose perfect-encoding gain is below
the target cannot reach it by operand-level work, no matter how good the solver.

Measured here for all 72 attention cases, using the evaluator's own scoring path
(`_attention`, `_score_details`, `encode/decode_standard_hif4`) and the current
root's calibrated states.  Also measured: the same ceilings for pairs and for all
three operands at once, which bounds every operand-level mechanism jointly.

Nothing is trained and no shard is run: calibration plus a handful of attention
forwards per case.
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


def main() -> int:
    torch.set_grad_enabled(False)
    solution = load_module(SOLUTION, "ceiling_solution")

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
            pairs = [
                v2._pair(entry[i].to(torch.float32)) for i in range(3)
            ]

            references = [v2.dequantize_nvfp4(*pair).to(torch.float32) for pair in pairs]
            standards = [
                v2.decode_standard_hif4(v2.encode_standard_hif4(value)).to(torch.float32)
                for value in references
            ]
            players = []
            for index, (params_api, pair, heads, state_name) in enumerate(
                (
                    (solution.hif4_dynamic_quantize_q, pairs[0], q_heads, "q_state"),
                    (solution.hif4_dynamic_quantize_k, pairs[1], kv_heads, "k_state"),
                    (solution.hif4_dynamic_quantize_v, pairs[2], kv_heads, "v_state"),
                )
            ):
                params = params_api(pair[0], pair[1], heads, head_dim, states[state_name])
                # Each operand decodes against its own reference shape: Q is
                # q_heads*head_dim wide, K and V are kv_heads*head_dim.
                shape = references[index].shape
                v2.validate_hif4_params(params, shape)
                players.append(
                    v2.dequantize_hif4(v2._cpu_params(params), shape).to(torch.float32)
                )

            def attention(qkv):
                return v2._attention(
                    *(t[None] for t in qkv), q_heads, kv_heads, head_dim
                )

            ref_out = attention(references)
            std_out = attention(standards)
            play_out = attention(players)

            mse_std = float((std_out - ref_out).square().mean())
            mse_play = float((play_out - ref_out).square().mean())

            # One operand perfect, the rest as played.
            variants = {}
            for index, name in enumerate(("q", "k", "v")):
                mixed = list(players)
                mixed[index] = references[index]
                variants[name] = float(
                    (attention(mixed) - ref_out).square().mean()
                )
            # Pairs and all three.
            for index_a, index_b, name in ((0, 1, "qk"), (0, 2, "qv"), (1, 2, "kv")):
                mixed = list(players)
                mixed[index_a] = references[index_a]
                mixed[index_b] = references[index_b]
                variants[name] = float((attention(mixed) - ref_out).square().mean())
            variants["all"] = 0.0

            def gain_of(mse: float) -> float:
                return 1.0 - mse / mse_std

            rows.append(
                {
                    "layer": layer,
                    "window": window,
                    "mse_standard": mse_std,
                    "mse_player": mse_play,
                    "gain_now": gain_of(mse_play),
                    **{f"gain_if_{k}_perfect": gain_of(v) for k, v in variants.items()},
                }
            )

    print(f"cases={len(rows)}")
    keys = ["gain_now"] + [f"gain_if_{k}_perfect" for k in ("q", "k", "v", "qk", "qv", "kv", "all")]
    header = f"{'layer':>5} " + " ".join(f"{k.replace('gain_if_','').replace('_perfect',''):>7}" for k in keys)
    print(header)
    summary = {}
    for layer in layers:
        sub = [r for r in rows if r["layer"] == layer]
        if not sub:
            continue
        line = f"{layer:>5} "
        for key in keys:
            value = statistics.mean(r[key] for r in sub)
            summary.setdefault(key, []).append(value)
            label = "now" if key == "gain_now" else key.replace("gain_if_", "").replace("_perfect", "")
            line += f"{value:>7.4f} "
        print(line + f"  (n={len(sub)})")
    overall = {key: statistics.mean(values) for key, values in summary.items()}
    print("\nALL 72:")
    for key in keys:
        label = "now" if key == "gain_now" else key.replace("gain_if_", "").replace("_perfect", "")
        print(f"  gain if {label:>4} perfect: {overall[key]:.4f}   (target 0.80)")

    payload = {
        "metric": "gain = 1 - mse_player/mse_standard; standard is the uncalibrated standard HiF4 codec",
        "cases": len(rows),
        "overall": overall,
        "rows": rows,
    }
    (HERE / "ceiling.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"\nwrote {HERE / 'ceiling.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
