"""Gate 0, corrected: the surrogate must be in the deployed gauge.

The first gate-0 run compared the attention loss's gradient with respect to Q_hat
against the gradient of `||Q_hat - Q_ref||^2`, where `Q_ref` is the *unrotated*
NVFP4 reference decode.  That is the wrong surrogate, and the user was right to
challenge it: the deployed pipeline is

    decode NVFP4 -> apply learned rotation -> (K: add learned center) -> encode HiF4

so `Q_hat` is rotated while `Q_ref` is not, and `Q_hat - Q_ref` is dominated by
the rotation rather than by the quantization error.  The direction it defines is
the rotation direction, not the encoding error's.

The quantity a Q-side rule can actually compute is the error **in the deployed
gauge**:

    ideal_deployed = rotate(Q_ref)                 (K: rotate(K_ref) + center)
    s_Q            = 2 (Q_hat - ideal_deployed) / numel

which is available to the API because the rotation and center live in its own
state.  This script re-runs the gate with that surrogate, and reports both so the
difference between the two readings is visible rather than asserted.

Same scope as before: forward plus one backward per window, real data, four
layers, no training and no shard.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import torch


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]

SOLUTION = ROOT / "solution.py"
PACK = ROOT / "artifacts/official_eval/cache/qwen3.5-4b-proxy-v2.pt"

LAYERS = (22, 15, 1, 0)


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    av = a.reshape(-1).float()
    bv = b.reshape(-1).float()
    denom = av.norm() * bv.norm()
    if float(denom) == 0.0:
        return 0.0
    return float((av @ bv) / denom)


def deployed_ideal(module, role, reference, state, heads, head_dim):
    """The same pipeline the codec runs on the input, minus the HiF4 step."""

    dense = reference
    rotation = state.get("learned_rotation")
    if rotation is not None:
        dense = module._a2_apply_group_rotation(
            dense, int(heads), rotation.to(device=dense.device, dtype=torch.float32)
        )
    center = state.get("learned_center")
    if center is not None:
        head_dim_c = int(dense.shape[-1]) // int(heads)
        lead = dense.shape[:-1]
        dense = (
            dense.reshape(*lead, int(heads), head_dim_c)
            + center.to(device=dense.device, dtype=torch.float32).reshape(
                *([1] * len(lead)), int(heads), head_dim_c
            )
        ).reshape(dense.shape)
    return dense


def main() -> int:
    torch.set_grad_enabled(True)
    module = load_module(SOLUTION, "gate0c_solution")

    sys.path.insert(0, str(ROOT / "evaluator"))
    import official_eval as v2  # noqa: PLC0415

    pack = torch.load(PACK, map_location="cpu", mmap=True, weights_only=False)
    q_heads = int(pack["q_heads"])
    kv_heads = int(pack["kv_heads"])
    head_dim = int(pack["head_dim"])
    device = torch.device("cpu")
    forward = module._a2_attention_forward

    results = []
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
        states = module.hif4_calibration_attention(windows, q_heads, kv_heads, head_dim)

        rows = []
        for window in range(len(pack["test_qkv"])):
            entry = pack["test_qkv"][window][layer]
            if entry is None:
                continue
            item = {
                role: tuple(t.to(device) for t in v2._pair(entry[i].to(torch.float32)))
                for i, role in enumerate(("q", "k", "v"))
            }
            hat, ref = {}, {}
            for role, heads in (("q", q_heads), ("k", kv_heads), ("v", kv_heads)):
                api = {
                    "q": module.hif4_dynamic_quantize_q,
                    "k": module.hif4_dynamic_quantize_k,
                    "v": module.hif4_dynamic_quantize_v,
                }[role]
                params = api(*item[role], heads, head_dim, states[role + "_state"])
                dense = module._dequantize_hif4(params).to(torch.float32)[None]
                dense.requires_grad_(True)
                hat[role] = dense
                ref[role] = module._dequantize_nvfp4_float32(*item[role]).to(
                    torch.float32
                )[None]

            actual = forward(hat["q"], hat["k"], hat["v"], q_heads, kv_heads, head_dim)
            target = forward(ref["q"], ref["k"], ref["v"], q_heads, kv_heads, head_dim)
            loss = (actual - target).square().mean()
            grads = torch.autograd.grad(
                loss, [hat["q"], hat["k"], hat["v"]], allow_unused=False
            )

            row = {"window": window, "loss": float(loss.item())}
            for index, (role, heads) in enumerate(
                (("q", q_heads), ("k", kv_heads), ("v", kv_heads))
            ):
                ideal = deployed_ideal(
                    module, role, ref[role], states[role + "_state"], heads, head_dim
                )
                raw = 2.0 * (hat[role] - ref[role]) / hat[role].numel()
                fixed = 2.0 * (hat[role] - ideal) / hat[role].numel()
                row[f"cos_raw_{role}"] = cosine(grads[index], raw)
                row[f"cos_gauge_{role}"] = cosine(grads[index], fixed)
                row[f"gap_abs_{role}"] = float(
                    (hat[role] - ideal).abs().mean().item()
                )
            rows.append(row)

        summary = {
            "layer": layer,
            "windows": len(rows),
            "mean_loss": sum(r["loss"] for r in rows) / len(rows),
            **{
                f"mean_cos_raw_{r}": sum(x[f"cos_raw_{r}"] for x in rows) / len(rows)
                for r in ("q", "k", "v")
            },
            **{
                f"mean_cos_gauge_{r}": sum(x[f"cos_gauge_{r}"] for x in rows) / len(rows)
                for r in ("q", "k", "v")
            },
        }
        results.append(summary)
        print(
            f"[layer {layer:>2}] n={summary['windows']} loss={summary['mean_loss']:.6f} | "
            f"Q raw {summary['mean_cos_raw_q']:+.3f} -> gauge {summary['mean_cos_gauge_q']:+.3f} | "
            f"K raw {summary['mean_cos_raw_k']:+.3f} -> gauge {summary['mean_cos_gauge_k']:+.3f} | "
            f"V raw {summary['mean_cos_raw_v']:+.3f} -> gauge {summary['mean_cos_gauge_v']:+.3f}",
            flush=True,
        )

    payload = {
        "question": (
            "With the surrogate taken in the deployed gauge, is the attention loss's "
            "gradient w.r.t. Q_hat aligned with a quantity the Q API can compute?"
        ),
        "results": results,
    }
    (HERE / "gate0-corrected.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    print(f"\nwrote {HERE / 'gate0-corrected.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
