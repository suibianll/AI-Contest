"""Isolate where the candidate's V rule diverges from the measured one.

Same parent parameters, same kernel (the candidate's own), same window.  Three
paths to the corrected V codes:

  PARENT   the parent's codes
  REBUILD  the diagnostic's greedy, rebuilt from the CANDIDATE'S OWN helpers
           (_vk_shifted / _vk_row_energy) -- isolates the building blocks
  CANDID   the candidate's `_vk_correct` -- isolates the orchestration

If REBUILD recovers the measured gain and CANDID does not, the bug is in
`_vk_correct`'s loop; if REBUILD already fails, the helpers differ.
"""

from __future__ import annotations

import importlib.util
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
    parent = load("iso_parent", ROOT / "solution.py")
    candidate = load("iso_candidate", HERE / "candidate" / "solution.py")

    sys.path.insert(0, str(ROOT / "evaluator"))
    import official_eval as v2  # noqa: PLC0415

    pack = torch.load(
        ROOT / "artifacts/official_eval/cache/qwen3.5-4b-proxy-v2.pt",
        map_location="cpu", mmap=True, weights_only=False,
    )
    q_heads = int(pack["q_heads"])
    kv_heads = int(pack["kv_heads"])
    dim = int(pack["head_dim"])
    group = q_heads // kv_heads
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    layer = int(os.environ.get("ISO_LAYER", "0"))
    window = int(os.environ.get("ISO_WINDOW", "3"))
    radius = candidate._VK_RADIUS
    nparam = candidate._VK_NPARAM

    def attention(q, k, v):
        return v2._attention(
            q.to(torch.float32)[None], k.to(torch.float32)[None], v.to(torch.float32)[None],
            q_heads, kv_heads, dim,
        )[0]

    def dense(params, shape):
        return v2.dequantize_hif4(v2._cpu_params(params), shape).to(torch.float32).to(device)

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
    states = candidate.hif4_calibration_attention(windows, q_heads, kv_heads, dim)
    kernel = states["v_state"]["vk_kernel"]

    entry = windows[window]
    shapes = [v2.dequantize_nvfp4(*entry[r]).shape for r in ("q", "k", "v")]
    pq = parent.hif4_dynamic_quantize_q(*entry["q"], q_heads, dim, states["q_state"])
    pk = parent.hif4_dynamic_quantize_k(*entry["k"], kv_heads, dim, states["k_state"])
    pv = parent.hif4_dynamic_quantize_v(*entry["v"], kv_heads, dim, states["v_state"])
    qd, kd = dense(pq, shapes[0]), dense(pk, shapes[1])
    v_ref = v2.dequantize_nvfp4(*entry["v"]).to(torch.float32).to(device)
    target = attention(qd, kd, v_ref)

    def loss(params):
        return float((attention(qd, kd, dense(params, shapes[2])) - target).square().mean())

    base = loss(pv)
    print(f"parent V-only loss = {base:.6e}")

    # --- REBUILD: the diagnostic's greedy, using the candidate's own helpers
    tokens = int(shapes[2][0])
    width = kv_heads * dim
    codes_i = (pv["mant"].to(torch.float64) * 4.0).reshape(tokens, width).clone()
    sign = pv["sign"].to(torch.float64)
    step = (
        pv["scale_factor"].to(torch.float64)
        * pv["scale_lv2"].to(torch.float64)
        * pv["scale_lv3"].to(torch.float64)
        / 4.0
    ).expand_as(pv["mant"].to(torch.float64))
    flat_step = (sign.reshape(tokens, width) * step.reshape(tokens, width)).to(device)
    sign_f = sign.reshape(tokens, width)
    movable = ((sign_f != 0) & (codes_i > 0) & (codes_i < 7)).to(device)
    codes_i = codes_i.to(device)
    v_ref3 = v_ref.reshape(tokens, kv_heads, dim).to(torch.float64)

    def to_params(codes):
        out = {k: v.clone() for k, v in pv.items()}
        m = (codes / 4.0).to(pv["mant"].dtype).reshape(pv["mant"].shape)
        out["mant"] = m
        out["sign"] = torch.where(
            m == 0, torch.zeros_like(pv["sign"]), pv["sign"]
        ).to(pv["sign"].dtype)
        return out

    def rebuild(codes):
        for _ in range(candidate._VK_SWEEPS):
            vals = (codes * flat_step).reshape(tokens, kv_heads, dim)
            resid = vals - v_ref3
            grad = torch.zeros_like(resid)
            qii = torch.zeros(tokens, dtype=torch.float64, device=device)
            for g in range(kv_heads):
                dg = resid[:, g, :]
                for j in range(group):
                    wg = kernel[g * group + j].to(torch.float64)
                    sig = candidate._vk_shifted(dg, wg, radius)
                    rev = torch.cat([wg[: candidate._VK_NBUCKET].flip(0), wg[candidate._VK_NBUCKET:]])
                    grad[:, g, :] += 2.0 * candidate._vk_shifted(sig, rev, radius)
                    qii += candidate._vk_row_energy(wg, tokens, radius, device)
            n_chan = kv_heads * dim
            fg = grad.reshape(tokens, n_chan)
            bg = torch.full((n_chan,), float("inf"), dtype=torch.float64, device=device)
            br = torch.zeros(n_chan, dtype=torch.long, device=device)
            bd = torch.zeros(n_chan, dtype=torch.float64, device=device)
            for direction in (1.0, -1.0):
                eps = direction * flat_step
                ch = fg * eps + (eps ** 2) * qii[:, None]
                ch = torch.where(movable, ch, torch.full_like(ch, float("inf")))
                gn, ar = ch.min(dim=0)
                tk = gn < bg
                bg = torch.where(tk, gn, bg)
                br = torch.where(tk, ar, br)
                bd = torch.where(tk, torch.full_like(bg, direction), bd)
            if not bool((bg < 0).any()):
                break
            imp = bg < 0
            cols = torch.arange(n_chan, device=device)
            upd = (codes[br, cols] + bd).clamp(0.0, 7.0)
            codes[br, cols] = torch.where(imp, upd, codes[br, cols])
        return codes

    rc = rebuild(codes_i.clone())
    print(f"REBUILD : loss={loss(to_params(rc)):.6e}  ratio={loss(to_params(rc))/base:+.4%}  "
          f"same-as-candidate={bool(torch.equal(rc, codes_i.clone()))}")

    cand = candidate._vk_correct(pv, kernel, *entry["v"], kv_heads, dim)
    cand_codes = (cand["mant"].to(torch.float64) * 4.0).reshape(tokens, width).to(device)
    print(f"CANDID  : loss={loss(cand):.6e}  ratio={loss(cand)/base:+.4%}")
    diff = int((cand_codes != rc).sum())
    print()
    print(f"codes differing between REBUILD and CANDID: {diff} / {cand_codes.numel()}")
    print(f"kernel params: radius={radius} nparam={nparam} shape={tuple(kernel.shape)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
