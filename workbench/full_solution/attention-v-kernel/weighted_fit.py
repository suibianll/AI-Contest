"""Residual-weighted kernel fit: does it close any of the 5.9x oracle gap?

The deployed kernel is fitted as the entrywise bucket mean of A, i.e. it
minimises sum (A - W)^2.  What the rule actually minimises is

    sum_t || (W - A) d_t ||^2 ,   d = V_hat - V_ref

and the two are equivalent only when the residual is i.i.d. across keys.  It is
not: HiF4 block scales make it strongly structured.  The fit that matches the
rule's objective is the residual-weighted bucket mean

    w_b = sum_{(t,k) in b} ||d_k||^2 * A[t,k]  /  sum_{(t,k) in b} ||d_k||^2

which is closed form, adds no deploy-time work, and needs no new state field --
only a different way of computing the same 32 numbers.

This measures, on the same window and with the same rule:

    PARENT    no correction
    KERNEL    the deployed entrywise fit
    WEIGHTED  the residual-weighted fit
    ORACLE    the true A (not deployable; the family's ceiling)
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
    root = load("wf_root", ROOT / "solution.py")
    cand = load("wf_cand", HERE / "candidate" / "solution.py")

    sys.path.insert(0, str(ROOT / "evaluator"))
    import official_eval as v2  # noqa: PLC0415

    pack = torch.load(
        ROOT / "artifacts/official_eval/cache/qwen3.5-4b-proxy-v2.pt",
        map_location="cpu", mmap=True, weights_only=False,
    )
    qh = int(pack["q_heads"]); kvh = int(pack["kv_heads"]); hd = int(pack["head_dim"])
    group = qh // kvh
    dev = torch.device(os.environ.get("WF_DEVICE", "cuda" if torch.cuda.is_available() else "cpu"))
    layer = int(os.environ.get("WF_LAYER", "0"))
    window = int(os.environ.get("WF_WINDOW", "3"))
    sweeps = int(os.environ.get("WF_SWEEPS", "6"))
    R = cand._VK_RADIUS; NB = cand._VK_NBUCKET; NP = cand._VK_NPARAM
    print(f"device={dev} layer={layer} window={window} sweeps={sweeps} radius={R}", flush=True)

    windows = [
        {
            role: tuple(t.to(dev) for t in v2._pair(pack["calibration_qkv"][s][layer][i].to(torch.float32)))
            for i, role in enumerate(("q", "k", "v"))
        }
        for s in range(len(pack["calibration_qkv"]))
    ]
    states = cand.hif4_calibration_attention(windows, qh, kvh, hd)
    deployed = states["v_state"].get("vk_kernel")

    def probs(qd, kd, h, g):
        return torch.softmax(qd[:, h, :] @ kd[:, g, :].transpose(-1, -2) / math.sqrt(hd), dim=-1)

    def bucket_mask(n):
        offs = torch.arange(n, device=dev)
        idx = offs[None, :] - offs[:, None]
        return torch.where(idx.abs() <= R, idx + R, torch.full_like(idx, NB))

    # --- fit both kernels on the fit windows, from deployed Q/K and the V residual
    tot_plain = torch.zeros(kvh, NP, dtype=torch.float64, device=dev)
    tot_w = torch.zeros(kvh, NP, dtype=torch.float64, device=dev)
    cnt = torch.zeros(NP, dtype=torch.float64, device=dev)
    for wi in (0, 1, 2):
        e = windows[wi]
        shapes = [v2.dequantize_nvfp4(*e[r]).shape for r in ("q", "k", "v")]
        pq = root.hif4_dynamic_quantize_q(*e["q"], qh, hd, states["q_state"])
        pk = root.hif4_dynamic_quantize_k(*e["k"], kvh, hd, states["k_state"])
        pv = root.hif4_dynamic_quantize_v(*e["v"], kvh, hd, states["v_state"])
        qd = v2.dequantize_hif4(v2._cpu_params(pq), shapes[0]).to(torch.float64).to(dev).reshape(-1, qh, hd)
        kd = v2.dequantize_hif4(v2._cpu_params(pk), shapes[1]).to(torch.float64).to(dev).reshape(-1, kvh, hd)
        vh = v2.dequantize_hif4(v2._cpu_params(pv), shapes[2]).to(torch.float64).to(dev)
        vr = v2.dequantize_nvfp4(*e["v"]).to(torch.float64).to(dev)
        n = int(shapes[0][0])
        mask = bucket_mask(n)
        resid = (vh - vr).reshape(n, kvh, hd)
        for g in range(kvh):
            ek = resid[:, g, :].square().sum(-1)              # (n,) per-key residual energy
            for h in range(g * group, (g + 1) * group):
                a = probs(qd, kd, h, g)
                for b in range(NP):
                    sel = mask == b
                    if not bool(sel.any()):
                        continue
                    wsel = ek.unsqueeze(0).expand(n, n)[sel]
                    tot_plain[g, b] += float(a[sel].sum())
                    tot_w[g, b] += float((a[sel] * wsel).sum())
                    if h == g * group:
                        cnt[b] += float(sel.sum())
    w_plain = (tot_plain / cnt.clamp_min(1.0)[None, :]).to(torch.float32)
    with torch.no_grad():
        wsum = torch.zeros(NP, dtype=torch.float64, device=dev)
        for wi in (0, 1, 2):
            e = windows[wi]
            shapes = [v2.dequantize_nvfp4(*e[r]).shape for r in ("q", "k", "v")]
            pq = root.hif4_dynamic_quantize_q(*e["q"], qh, hd, states["q_state"])
            pk = root.hif4_dynamic_quantize_k(*e["k"], kvh, hd, states["k_state"])
            pv = root.hif4_dynamic_quantize_v(*e["v"], kvh, hd, states["v_state"])
            vh = v2.dequantize_hif4(v2._cpu_params(pv), shapes[2]).to(torch.float64).to(dev)
            vr = v2.dequantize_nvfp4(*e["v"]).to(torch.float64).to(dev)
            n = int(shapes[0][0]); mask = bucket_mask(n)
            resid = (vh - vr).reshape(n, kvh, hd)
            ek = resid.square().sum(-1).sum(-1)               # (n,) summed over groups
            for b in range(NP):
                sel = mask == b
                if bool(sel.any()):
                    wsum[b] += float(ek.unsqueeze(0).expand(n, n)[sel].sum())
    w_weighted = (tot_w / wsum.clamp_min(1e-30)[None, :]).to(torch.float32)
    print(f"plain   vs weighted kernel max|gap| = {float((w_plain-w_weighted).abs().max()):.3e}", flush=True)
    if deployed is not None:
        d = deployed.to(device=w_plain.device, dtype=w_plain.dtype)
        print(f"deployed vs replain   max|gap| = {float((w_plain-d).abs().max()):.3e}", flush=True)

    # --- evaluate on the eval window
    e = windows[window]
    shapes = [v2.dequantize_nvfp4(*e[r]).shape for r in ("q", "k", "v")]
    pq = root.hif4_dynamic_quantize_q(*e["q"], qh, hd, states["q_state"])
    pk = root.hif4_dynamic_quantize_k(*e["k"], kvh, hd, states["k_state"])
    pv = root.hif4_dynamic_quantize_v(*e["v"], kvh, hd, states["v_state"])
    qd = v2.dequantize_hif4(v2._cpu_params(pq), shapes[0]).to(torch.float64).to(dev)
    kd = v2.dequantize_hif4(v2._cpu_params(pk), shapes[1]).to(torch.float64).to(dev)
    vref = v2.dequantize_nvfp4(*e["v"]).to(torch.float64).to(dev)
    qd3 = qd.reshape(-1, qh, hd); kd3 = kd.reshape(-1, kvh, hd)
    tokens = int(shapes[2][0]); width = kvh * hd

    def attention(q, k, v):
        return v2._attention(q.to(torch.float32)[None], k.to(torch.float32)[None],
                             v.to(torch.float32)[None], qh, kvh, hd)[0]

    target = attention(qd, kd, vref)
    A = [probs(qd3, kd3, h, h // group) for h in range(qh)]

    sign = pv["sign"].to(torch.float64)
    step = (pv["scale_factor"].to(torch.float64) * pv["scale_lv2"].to(torch.float64)
            * pv["scale_lv3"].to(torch.float64) / 4.0).expand_as(pv["mant"].to(torch.float64))
    fs = (sign * step).reshape(tokens, width).to(dev)
    c0 = (pv["mant"].to(torch.float64) * 4.0).reshape(tokens, width).to(dev)
    movable = ((sign.reshape(tokens, width) != 0) & (c0 > 0) & (c0 < 7)).to(dev)

    def decode(codes):
        out = {k: v.clone() for k, v in pv.items()}
        m = (codes / 4.0).to(pv["mant"].dtype).reshape(pv["mant"].shape)
        out["mant"] = m
        out["sign"] = torch.where(m == 0, torch.zeros_like(pv["sign"]), pv["sign"]).to(pv["sign"].dtype)
        return v2.dequantize_hif4(v2._cpu_params(out), shapes[2]).to(torch.float64).to(dev)

    def loss(c):
        return float((attention(qd, kd, decode(c)) - target).square().mean())

    base = loss(c0)
    print(f"parent V-only loss = {base:.6e}", flush=True)

    def run(kernel, oracle):
        codes = c0.clone()
        for _ in range(sweeps):
            vals = (codes * fs).reshape(tokens, kvh, hd)
            resid = vals - vref.reshape(tokens, kvh, hd)
            grad = torch.zeros_like(resid)
            qii = torch.zeros(tokens, dtype=torch.float64, device=dev)
            for h in range(qh):
                g = h // group
                dg = resid[:, g, :]
                if oracle:
                    a = A[h]
                    grad[:, g, :] += 2.0 * (a.transpose(-1, -2) @ (a @ dg))
                    qii += (a * a).sum(0)
                else:
                    wg = kernel[g].to(dev, torch.float64)
                    sig = cand._vk_shifted(dg, wg, R)
                    rev = torch.cat([wg[:NB].flip(0), wg[NB:]])
                    grad[:, g, :] += 2.0 * cand._vk_shifted(sig, rev, R)
                    qii += cand._vk_row_energy(wg, tokens, R, dev)
            fg = grad.reshape(tokens, width)
            bg = torch.full((width,), float("inf"), dtype=torch.float64, device=dev)
            br = torch.zeros(width, dtype=torch.long, device=dev)
            bd = torch.zeros(width, dtype=torch.float64, device=dev)
            for k in (-1.0, 1.0):
                cc = codes + k
                legal = movable & (cc >= 0) & (cc <= 7)
                eps = k * fs
                ch = fg * eps + (eps ** 2) * qii[:, None]
                ch = torch.where(legal, ch, torch.full_like(ch, float("inf")))
                gn_, ar = ch.min(dim=0)
                tk = gn_ < bg
                bg = torch.where(tk, gn_, bg); br = torch.where(tk, ar, br)
                bd = torch.where(tk, torch.full_like(bg, k), bd)
            if not bool((bg < 0).any()):
                break
            imp = bg < 0
            cols = torch.arange(width, device=dev)
            upd = (codes[br, cols] + bd).clamp(0.0, 7.0)
            codes[br, cols] = torch.where(imp, upd, codes[br, cols])
        return codes

    for name, ker, orc in (("KERNEL  ", w_plain, False), ("WEIGHTED", w_weighted, False),
                           ("ORACLE  ", None, True)):
        c = run(ker, orc)
        l = loss(c)
        print(f"{name}: loss={l:.6e}  change={(l/base-1)*100:+.4f}%  moved={int((c != c0).sum())}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
