"""The V rule's ceiling: what if the weighting were the TRUE attention?

VK-1 measured the fitted Toeplitz kernel against A and found them 0.80 of A's
own response magnitude apart.  That could be what caps the V-side gain at ~1%.
This runs the SAME rule -- same +/-1 closed-form greedy, same sweeps -- with the
weighting set to

  KERNEL   the compiled Toeplitz stand-in (what the candidate deploys)
  ORACLE   the true per-head A, a full T x T matrix (NOT deployable)

and scores both with the real, validating decoder against the real attention.
If ORACLE is barely better, the kernel is not the bottleneck and the rule shape
is; if ORACLE is far better, the kernel's fit error is what to attack next.

FOR Q:  g = 2 sum_h A_h^T (A_h d) ,  Q_ii = sum_h (A_h^T A_h)_ii
(For the Toeplitz kernel the same expressions hold with A_h replaced by W_h.)

CPU or GPU.
"""

from __future__ import annotations

import importlib.util
import math
import os
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def main() -> int:
    torch.set_num_threads(4)
    torch.set_grad_enabled(False)
    root = load("oc_root", ROOT / "solution.py")
    cand = load("oc_cand", HERE / "candidate" / "solution.py")

    sys.path.insert(0, str(ROOT / "evaluator"))
    import official_eval as v2  # noqa: PLC0415

    pack = torch.load(
        ROOT / "artifacts/official_eval/cache/qwen3.5-4b-proxy-v2.pt",
        map_location="cpu", mmap=True, weights_only=False,
    )
    qh = int(pack["q_heads"]); kvh = int(pack["kv_heads"]); hd = int(pack["head_dim"])
    group = qh // kvh
    dev = torch.device(os.environ.get("OC_DEVICE", "cuda" if torch.cuda.is_available() else "cpu"))
    layer = int(os.environ.get("OC_LAYER", "0"))
    window = int(os.environ.get("OC_WINDOW", "3"))
    sweeps = int(os.environ.get("OC_SWEEPS", "6"))
    print(f"device={dev} layer={layer} window={window} sweeps={sweeps}", flush=True)

    windows = [
        {
            role: tuple(t.to(dev) for t in v2._pair(pack["calibration_qkv"][s][layer][i].to(torch.float32)))
            for i, role in enumerate(("q", "k", "v"))
        }
        for s in range(len(pack["calibration_qkv"]))
    ]
    states = cand.hif4_calibration_attention(windows, qh, kvh, hd)
    kernel = states["v_state"].get("vk_kernel")

    e = windows[window]
    shapes = [v2.dequantize_nvfp4(*e[r]).shape for r in ("q", "k", "v")]
    pq = root.hif4_dynamic_quantize_q(*e["q"], qh, hd, states["q_state"])
    pk = root.hif4_dynamic_quantize_k(*e["k"], kvh, hd, states["k_state"])
    pv = root.hif4_dynamic_quantize_v(*e["v"], kvh, hd, states["v_state"])
    qd = v2.dequantize_hif4(v2._cpu_params(pq), shapes[0]).to(torch.float64).to(dev)
    kd = v2.dequantize_hif4(v2._cpu_params(pk), shapes[1]).to(torch.float64).to(dev)
    v_ref = v2.dequantize_nvfp4(*e["v"]).to(torch.float64).to(dev)
    tokens = int(shapes[2][0]); width = kvh * hd

    def attention(q, k, v):
        return v2._attention(q.to(torch.float32)[None], k.to(torch.float32)[None],
                             v.to(torch.float32)[None], qh, kvh, hd)[0]

    target = attention(qd, kd, v_ref)
    qh_ = qd.reshape(-1, qh, hd); kh_ = kd.reshape(-1, kvh, hd)

    A = []
    for h in range(qh):
        logits = qh_[:, h, :] @ kh_[:, h // group, :].transpose(-1, -2) / math.sqrt(hd)
        A.append(torch.softmax(logits, dim=-1))

    sign = pv["sign"].to(torch.float64)
    step = (pv["scale_factor"].to(torch.float64) * pv["scale_lv2"].to(torch.float64)
            * pv["scale_lv3"].to(torch.float64) / 4.0).expand_as(pv["mant"].to(torch.float64))
    flat_step = (sign * step).reshape(tokens, width).to(dev)
    code0 = (pv["mant"].to(torch.float64) * 4.0).reshape(tokens, width).to(dev)
    movable = ((sign.reshape(tokens, width) != 0) & (code0 > 0) & (code0 < 7)).to(dev)

    def decode(codes):
        out = {k: v.clone() for k, v in pv.items()}
        m = (codes / 4.0).to(pv["mant"].dtype).reshape(pv["mant"].shape)
        out["mant"] = m
        out["sign"] = torch.where(m == 0, torch.zeros_like(pv["sign"]), pv["sign"]).to(pv["sign"].dtype)
        return v2.dequantize_hif4(v2._cpu_params(out), shapes[2]).to(torch.float64).to(dev)

    def loss(codes):
        return float((attention(qd, kd, decode(codes)) - target).square().mean())

    base = loss(code0)
    print(f"parent V-only loss = {base:.6e}", flush=True)

    def run(use_oracle):
        codes = code0.clone()
        for _ in range(sweeps):
            vals = (codes * flat_step).reshape(tokens, kvh, hd)
            resid = vals - v_ref.reshape(tokens, kvh, hd)
            grad = torch.zeros_like(resid)
            qii = torch.zeros(tokens, dtype=torch.float64, device=dev)
            for h in range(qh):
                g = h // group
                dg = resid[:, g, :]
                if use_oracle:
                    a = A[h]
                    grad[:, g, :] += 2.0 * (a.transpose(-1, -2) @ (a @ dg))
                    qii += (a * a).sum(0)
                else:
                    wg = kernel[g].to(dev, torch.float64) if kernel is not None else None
                    if wg is None:
                        return codes
                    sig = cand._vk_shifted(dg, wg, cand._VK_RADIUS)
                    rev = torch.cat([wg[: cand._VK_NBUCKET].flip(0), wg[cand._VK_NBUCKET:]])
                    grad[:, g, :] += 2.0 * cand._vk_shifted(sig, rev, cand._VK_RADIUS)
                    qii += cand._vk_row_energy(wg, tokens, cand._VK_RADIUS, dev)
            n_chan = width
            fg = grad.reshape(tokens, n_chan)
            bg = torch.full((n_chan,), float("inf"), dtype=torch.float64, device=dev)
            br = torch.zeros(n_chan, dtype=torch.long, device=dev)
            bd = torch.zeros(n_chan, dtype=torch.float64, device=dev)
            for k in (-1.0, 1.0):
                cand_c = codes + k
                legal = movable & (cand_c >= 0) & (cand_c <= 7)
                eps = k * flat_step
                ch = fg * eps + (eps ** 2) * qii[:, None]
                ch = torch.where(legal, ch, torch.full_like(ch, float("inf")))
                gn, ar = ch.min(dim=0)
                tk = gn < bg
                bg = torch.where(tk, gn, bg)
                br = torch.where(tk, ar, br)
                bd = torch.where(tk, torch.full_like(bg, k), bd)
            if not bool((bg < 0).any()):
                break
            imp = bg < 0
            cols = torch.arange(n_chan, device=dev)
            upd = (codes[br, cols] + bd).clamp(0.0, 7.0)
            codes[br, cols] = torch.where(imp, upd, codes[br, cols])
        return codes

    for name, flag in (("KERNEL", False), ("ORACLE", True)):
        c = run(flag)
        moved = int((c != code0).sum())
        l = loss(c)
        print(f"{name}: loss={l:.6e}  change={(l/base-1)*100:+.4f}%  moved={moved}", flush=True)
    print()
    print("(ORACLE uses the true A and is NOT deployable; it bounds the rule family,")
    print(" not the kernel.  A large ORACLE/KERNEL gap points at the kernel's fit;")
    print(" a small one points at the rule shape.)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
