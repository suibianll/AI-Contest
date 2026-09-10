"""Gate 0: can a per-API rule express the joint optimum's Q (or K) component?

The plan's first gate asks whether the "Q/K joint output-residual" information can
be compressed into a rule that each dynamic API can execute from its own input
plus compiled state.  The objective itself is a calibration-time object, so the
question is not "can the loss be evaluated at dynamic time" -- it cannot, and it
does not need to be.  The question is narrower and testable:

    is the attention loss's gradient with respect to Q_hat aligned with the
    gradient of a Q-operand quantity that the Q API *can* compute?

The Q API receives `q_quant, q_scale, q_num_heads, head_dim, q_state`.  From those
it can build its own reference decode `Q_ref = dequantize_nvfp4(q_quant,q_scale)`
and its own deployed decode `Q_hat`; what it cannot build is K, V, the softmax,
or the attention output.  So the separable surrogate it can actually evaluate is
its own operand error `||Q_hat - Q_ref||^2` (or a calibrated reweighting of it).

This script measures, on real windows of the worst layers, the cosine similarity
between

    g_Q = d L_attention / d Q_hat          (needs K, V, softmax -- not computable)
    s_Q = d ||Q_hat - Q_ref||^2 / d Q_hat  (computable by the Q API alone)

and the same pair for K, plus the same comparison against V for context.

Reading the result:

  * cos close to 1  -- the joint structure adds little to the Q direction, and a
    separable per-API rule can plausibly carry it.  Gate passes, main line stands.
  * cos away from 1 -- the Q-optimal direction is organised by K/V/softmax, and no
    function of Q alone can reproduce it.  Gate fails and the main line is
    API_INCOMPATIBLE in substance, not merely in form.

Nothing is trained here and no shard is run: this is forward plus one backward per
window.  The number is a direction alignment, not a score.
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


def main() -> int:
    torch.set_grad_enabled(True)
    module = load_module(SOLUTION, "gate0_solution")

    sys.path.insert(0, str(ROOT / "evaluator"))
    import official_eval as v2  # noqa: PLC0415

    pack = torch.load(PACK, map_location="cpu", mmap=True, weights_only=False)
    q_heads = int(pack["q_heads"])
    kv_heads = int(pack["kv_heads"])
    head_dim = int(pack["head_dim"])
    device = torch.device("cpu")

    # The deployed attention path, exactly as the A-GR1 gate builds it.
    forward = module._a2_attention_forward
    decode_hif4 = module._dequantize_hif4
    decode_nvfp4 = module._dequantize_nvfp4_float32

    # The states must come from a real calibration; build them once per layer.
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

        layer_rows = []
        for window in range(len(pack["test_qkv"])):
            entry = pack["test_qkv"][window][layer]
            if entry is None:
                continue
            item = {
                role: tuple(t.to(device) for t in v2._pair(entry[i].to(torch.float32)))
                for i, role in enumerate(("q", "k", "v"))
            }

            hat = {}
            ref = {}
            for role, heads in (("q", q_heads), ("k", kv_heads), ("v", kv_heads)):
                api = {
                    "q": module.hif4_dynamic_quantize_q,
                    "k": module.hif4_dynamic_quantize_k,
                    "v": module.hif4_dynamic_quantize_v,
                }[role]
                params = api(*item[role], heads, head_dim, states[role + "_state"])
                dense = decode_hif4(params).to(torch.float32)[None]
                dense.requires_grad_(True)
                hat[role] = dense
                ref[role] = decode_nvfp4(*item[role]).to(torch.float32)[None]

            actual = forward(hat["q"], hat["k"], hat["v"], q_heads, kv_heads, head_dim)
            target = forward(ref["q"], ref["k"], ref["v"], q_heads, kv_heads, head_dim)
            loss = (actual - target).square().mean()

            grads = torch.autograd.grad(loss, [hat["q"], hat["k"], hat["v"]], allow_unused=False)
            self_grads = [
                2.0 * (hat[role] - ref[role]) / hat[role].numel()
                for role in ("q", "k", "v")
            ]

            row = {
                "window": window,
                "loss": float(loss.item()),
                "cos_q": cosine(grads[0], self_grads[0]),
                "cos_k": cosine(grads[1], self_grads[1]),
                "cos_v": cosine(grads[2], self_grads[2]),
                "g_q_norm": float(grads[0].norm().item()),
                "s_q_norm": float(self_grads[0].norm().item()),
                "g_k_norm": float(grads[1].norm().item()),
                "s_k_norm": float(self_grads[1].norm().item()),
            }
            layer_rows.append(row)

        if not layer_rows:
            continue
        summary = {
            "layer": layer,
            "windows": len(layer_rows),
            "mean_loss": sum(r["loss"] for r in layer_rows) / len(layer_rows),
            "mean_cos_q": sum(r["cos_q"] for r in layer_rows) / len(layer_rows),
            "mean_cos_k": sum(r["cos_k"] for r in layer_rows) / len(layer_rows),
            "mean_cos_v": sum(r["cos_v"] for r in layer_rows) / len(layer_rows),
            "rows": layer_rows,
        }
        results.append(summary)
        print(
            f"[layer {layer:>2}] windows={summary['windows']} loss={summary['mean_loss']:.6f} | "
            f"cos(g_Q, self_Q) = {summary['mean_cos_q']:+.3f} | "
            f"cos(g_K, self_K) = {summary['mean_cos_k']:+.3f} | "
            f"cos(g_V, self_V) = {summary['mean_cos_v']:+.3f}",
            flush=True,
        )

    payload = {
        "question": (
            "Is the attention loss's gradient w.r.t. Q_hat aligned with the Q-operand "
            "error gradient, which the Q API can compute from its own input?"
        ),
        "reading": (
            "cos near 1 -> a separable per-API rule can plausibly carry the direction; "
            "cos away from 1 -> the Q-optimal direction is organised by K/V/softmax and "
            "no function of Q alone reproduces it."
        ),
        "results": results,
    }
    (HERE / "gate0.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"\nwrote {HERE / 'gate0.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
