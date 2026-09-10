"""Where does the attention error actually live?  Q-side, K-side, or V-side.

The VK candidate's gate compares FULL attention losses, and a V-side mechanism
measured at -0.99% on the V-only objective moved the full loss by 0.026%.  Those
two numbers are both right and they are not the same quantity, so before any
more work goes into V, this measures the split directly:

for each layer and window, with everything else at its deployed (player) value,
swap ONE operand back to its NVFP4 reference and read the resulting loss:

    all_deployed    Attn(Qh, Kh, Vh)  vs  Attn(Qr, Kr, Vr)      the scored error
    V_only          Attn(Qh, Kh, Vr)  vs  the same target
    Q_only          Attn(Qr, Kh, Vh)  vs  the same target
    K_only          Attn(Qh, Kr, Vh)  vs  the same target

and the complementary "how much is left if this operand were perfect":

    Q_perfect       Attn(Qr, Kh, Vh)
    K_perfect       Attn(Qh, Kr, Vh)
    V_perfect       Attn(Qh, Kh, Vr)

The reference is the NVFP4 decode, exactly as the evaluator builds it.  These
are attribution readings, not bounds -- making one operand lossless is not a
reachable state -- but they say which side a mechanism has to move to matter.

CPU or GPU.  Small: one pass per layer.
"""

from __future__ import annotations

import importlib.util
import json
import os
import statistics
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
SOLUTION = ROOT / "solution.py"

LAYERS = tuple(int(x) for x in os.environ.get("DEX_LAYERS", "0,1,5,8,15,22").split(","))
WINDOWS = tuple(int(x) for x in os.environ.get("DEX_WINDOWS", "0,1,2,3,4").split(","))


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def main() -> int:
    torch.set_num_threads(4)
    torch.set_grad_enabled(False)
    solution = load("dex_solution", SOLUTION)

    sys.path.insert(0, str(ROOT / "evaluator"))
    import official_eval as v2  # noqa: PLC0415

    pack = torch.load(
        ROOT / "artifacts/official_eval/cache/qwen3.5-4b-proxy-v2.pt",
        map_location="cpu", mmap=True, weights_only=False,
    )
    q_heads = int(pack["q_heads"])
    kv_heads = int(pack["kv_heads"])
    dim = int(pack["head_dim"])
    device = torch.device(os.environ.get("DEX_DEVICE", "cuda" if torch.cuda.is_available() else "cpu"))
    print(f"device={device}", flush=True)

    def attention(q, k, v):
        return v2._attention(
            q.to(torch.float32)[None], k.to(torch.float32)[None], v.to(torch.float32)[None],
            q_heads, kv_heads, dim,
        )[0]

    def dense(params, shape):
        return v2.dequantize_hif4(v2._cpu_params(params), shape).to(torch.float32).to(device)

    def mse(a, b):
        return float((a - b).square().mean())

    rows = []
    for layer in LAYERS:
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
        states = solution.hif4_calibration_attention(windows, q_heads, kv_heads, dim)
        for w_index in WINDOWS:
            entry = windows[w_index]
            shapes = [v2.dequantize_nvfp4(*entry[r]).shape for r in ("q", "k", "v")]
            pq = solution.hif4_dynamic_quantize_q(*entry["q"], q_heads, dim, states["q_state"])
            pk = solution.hif4_dynamic_quantize_k(*entry["k"], kv_heads, dim, states["k_state"])
            pv = solution.hif4_dynamic_quantize_v(*entry["v"], kv_heads, dim, states["v_state"])
            qh, kh, vh = (dense(p, s) for p, s in zip((pq, pk, pv), shapes))
            qr = v2.dequantize_nvfp4(*entry["q"]).to(torch.float32).to(device)
            kr = v2.dequantize_nvfp4(*entry["k"]).to(torch.float32).to(device)
            vr = v2.dequantize_nvfp4(*entry["v"]).to(torch.float32).to(device)
            target = attention(qr, kr, vr)

            all_dep = mse(attention(qh, kh, vh), target)
            row = {
                "layer": layer,
                "window": w_index,
                "tokens": int(shapes[0][0]),
                "all_deployed": all_dep,
                # swap ONE operand back to its reference
                "loss_q_only": mse(attention(qh, kr, vr), target),
                "loss_k_only": mse(attention(qr, kh, vr), target),
                "loss_v_only": mse(attention(qr, kr, vh), target),
                # keep ONE operand deployed, everything else perfect
                "loss_q_perfect": mse(attention(qr, kh, vh), target),
                "loss_k_perfect": mse(attention(qh, kr, vh), target),
                "loss_v_perfect": mse(attention(qh, kh, vr), target),
            }
            rows.append(row)
            print(
                f"[L{layer:>2}/w{w_index} T={row['tokens']:>4}] all={all_dep:.4e} | "
                f"only-Q={row['loss_q_only']:.3e} only-K={row['loss_k_only']:.3e} only-V={row['loss_v_only']:.3e} | "
                f"Q-perfect={row['loss_q_perfect']:.3e} K-perfect={row['loss_k_perfect']:.3e} "
                f"V-perfect={row['loss_v_perfect']:.3e}",
                flush=True,
            )

    print()
    print("mean over windows, as a fraction of the all-deployed loss:")
    for key, name in (
        ("loss_q_only", "only Q deployed"), ("loss_k_only", "only K deployed"),
        ("loss_v_only", "only V deployed"),
        ("loss_q_perfect", "Q perfect (K,V deployed)"),
        ("loss_k_perfect", "K perfect (Q,V deployed)"),
        ("loss_v_perfect", "V perfect (Q,K deployed)"),
    ):
        vals = [r[key] / r["all_deployed"] for r in rows if r["all_deployed"] > 0]
        print(f"  {name:<26} {statistics.mean(vals):.4f}   range=[{min(vals):.4f}, {max(vals):.4f}]")
    print()
    print("per layer, 'V perfect / all deployed' and 'Q perfect / all deployed':")
    for layer in LAYERS:
        sub = [r for r in rows if r["layer"] == layer and r["all_deployed"] > 0]
        if not sub:
            continue
        vp = statistics.mean(r["loss_v_perfect"] / r["all_deployed"] for r in sub)
        qp = statistics.mean(r["loss_q_perfect"] / r["all_deployed"] for r in sub)
        kp = statistics.mean(r["loss_k_perfect"] / r["all_deployed"] for r in sub)
        print(f"  L{layer:>2}  Q {qp:.4f}   K {kp:.4f}   V {vp:.4f}")

    (HERE / "decompose-error.json").write_text(
        json.dumps({"layers": list(LAYERS), "windows": list(WINDOWS), "rows": rows}, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {HERE / 'decompose-error.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
