"""VK-3: kernel-weighted V code selection, measured on the true output MSE.

`v_error_balance.py` chose V codes to shrink the per-channel summed error -- the
UNIFORM kernel.  VK-1/VK-2 characterised what that costs: the uniform weighting
understates the real damage by ~17x in magnitude, and once perturbation size is
controlled for it carries almost no shape information, while a fitted
relative-position kernel keeps a positive shape component.

So this card changes exactly one thing -- the weighting -- and measures the one
thing that matters:

    UNIFORM   min sum_c sum_t (sum_k (1/T) d[k,c])^2                  (+0.0038 measured)
    KERNEL    min sum_g sum_{h in g} sum_t (sum_k w_h[t-k] d[k,c])^2   this card

where d = V_hat_hiF4 - V_ref_nvfp4 and w_h is fitted on FIT windows only.
d is available at deploy time: the V API receives the NVFP4 pair and produces the
codes, so it knows both ends.

The kernel is 32 parameters per Q head: 31 banded weights for |k-t| <= 15 plus
one shared weight for every larger offset.  The banded part is computed with
`unfold` on a radius-padded tensor, so the action costs O(T * C * 32) and a T x T
matrix is never materialised.

Independent check first: the banded action is compared against an explicitly
built Toeplitz matrix on every window, and the gap is printed before any gain
number.  (Both helpers were wrong in the first draft -- the unfold window was
`delta[t .. t+2R]` instead of `delta[t-R .. t+R]`, and the far bucket collided
with offset +R.  The check exists because that is exactly the failure that a
plausible-looking gain number would have hidden.)

CPU or GPU; smoke first:

    VK3_LAYERS=0 VK3_EVAL_WINDOWS=0,1 VK3_TOKENS=128 VK3_SWEEPS=2 python vk3_rule.py
"""

from __future__ import annotations

import importlib.util
import json
import math
import os
import statistics
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
SOLUTION = ROOT / "solution.py"

LAYERS = tuple(int(x) for x in os.environ.get("VK3_LAYERS", "0,1,5,8,15,22").split(","))
FIT_WINDOWS = (0, 1, 2)
EVAL_WINDOWS = tuple(
    int(x) for x in os.environ.get("VK3_EVAL_WINDOWS", "3,4").split(",")
)
FIT_TOKENS = int(os.environ.get("VK3_FIT_TOKENS", "48"))
# Calibration windows hold 10 / 128 / 512 / 1024 / 1024 tokens.  The default
# eval windows (3, 4) are the 1024-token ones, so the cap must admit them for
# the full run; smoke on windows 0/1 with a small cap instead.
MAX_TOKENS = int(os.environ.get("VK3_TOKENS", "1024"))
SWEEPS = int(os.environ.get("VK3_SWEEPS", "6"))
# The fitted kernels put their mass within |r| <= 3 (VK-1), so the support
# radius is a cost knob, not an accuracy knob: the deploy cost is O(T*C*(2R+1)).
KERNEL_RADIUS = int(os.environ.get("VK3_RADIUS", "15"))
NBUCKET = 2 * KERNEL_RADIUS + 1        # offsets -R..+R
NFAR = NBUCKET                          # index of the shared far bucket
NPARAM = NBUCKET + 1                    # 32


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def pick_tokens(total: int, count: int) -> list:
    if total <= count:
        return list(range(total))
    return sorted({int(round(x)) for x in torch.linspace(0, total - 1, count).tolist()})


def main() -> int:
    torch.set_num_threads(4)
    torch.set_grad_enabled(False)
    solution = load("vk3_solution", SOLUTION)

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
    device = torch.device(os.environ.get("VK3_DEVICE", "cuda" if torch.cuda.is_available() else "cpu"))
    print(f"device={device}", flush=True)

    def dense(params, shape):
        return v2.dequantize_hif4(v2._cpu_params(params), shape).to(torch.float64).to(device)

    def attention(q, k, v):
        return v2._attention(
            q.to(torch.float32)[None], k.to(torch.float32)[None], v.to(torch.float32)[None],
            q_heads, kv_heads, dim,
        )[0]

    def probs_for(qd, kd, h, g):
        logits = qd[:, h, :] @ kd[:, g, :].transpose(-1, -2) / math.sqrt(dim)
        return torch.softmax(logits, dim=-1)

    def banded(weights, delta):
        """(T, C) -> (T, C): the Toeplitz action of a 32-parameter kernel.

        weights[0..30] are offsets -15..+15; weights[31] is shared by every
        larger offset.  Row t of the banded part needs delta[t-15 .. t+15], so
        the tensor is padded by RADIUS on BOTH sides and then unfolded; padding
        only on the right would have produced delta[t .. t+2R].
        """

        near = weights[:NBUCKET]
        far = weights[NFAR]
        pad = torch.zeros(KERNEL_RADIUS, delta.shape[1], dtype=delta.dtype, device=delta.device)
        padded = torch.cat([pad, delta, pad], dim=0)               # (T + 2R, C)
        windows = padded.unfold(0, NBUCKET, 1)                     # (T, C, 2R+1)
        near_sum = windows.sum(-1)                                 # (T, C)
        total = delta.sum(0, keepdim=True)                         # (1, C)
        return torch.einsum("tcr,r->tc", windows, near) + far * (total - near_sum)

    def toeplitz(weights, tokens):
        idx = torch.full((tokens, tokens), NFAR, dtype=torch.long)
        for t in range(tokens):
            for k in range(tokens):
                off = k - t
                if abs(off) <= KERNEL_RADIUS:
                    idx[t, k] = off + KERNEL_RADIUS
        return weights[idx.to(device)]

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

        # fit: w[h, b] = mean of A_h over the entries in bucket b (fit windows only)
        totals = torch.zeros(q_heads, NPARAM, dtype=torch.float64, device=device)
        counts = torch.zeros(NPARAM, dtype=torch.float64, device=device)
        for index in FIT_WINDOWS:
            entry = windows[index]
            shapes = [v2.dequantize_nvfp4(*entry[r]).shape for r in ("q", "k", "v")]
            pq = solution.hif4_dynamic_quantize_q(*entry["q"], q_heads, dim, states["q_state"])
            pk = solution.hif4_dynamic_quantize_k(*entry["k"], kv_heads, dim, states["k_state"])
            qd = dense(pq, shapes[0]).reshape(-1, q_heads, dim)
            kd = dense(pk, shapes[1]).reshape(-1, kv_heads, dim)
            keep = pick_tokens(int(shapes[0][0]), FIT_TOKENS)
            idxm = toeplitz(
                torch.arange(NPARAM, dtype=torch.float64, device=device).long(), int(shapes[0][0])
            )[keep]
            for h in range(q_heads):
                p = probs_for(qd, kd, h, h // group)[keep]
                for b in range(NPARAM):
                    sel = idxm == b
                    if bool(sel.any()):
                        totals[h, b] += p[sel].sum()
                        if h == 0:
                            counts[b] += float(sel.sum())
        w = (totals / counts.clamp_min(1.0)[None, :]).contiguous()
        if os.environ.get("VK3_DUMP"):
            torch.save(w.cpu(), os.environ["VK3_DUMP"])
            print(f"dumped kernel to {os.environ['VK3_DUMP']}", flush=True)
            return 0

        for w_index in EVAL_WINDOWS:
            entry = windows[w_index]
            shapes = [v2.dequantize_nvfp4(*entry[r]).shape for r in ("q", "k", "v")]
            tokens = int(shapes[0][0])
            if tokens > MAX_TOKENS:
                continue
            pq = solution.hif4_dynamic_quantize_q(*entry["q"], q_heads, dim, states["q_state"])
            pk = solution.hif4_dynamic_quantize_k(*entry["k"], kv_heads, dim, states["k_state"])
            pv = solution.hif4_dynamic_quantize_v(*entry["v"], kv_heads, dim, states["v_state"])
            qd = dense(pq, shapes[0]).reshape(-1, q_heads, dim)
            kd = dense(pk, shapes[1]).reshape(-1, kv_heads, dim)
            v_ref = v2.dequantize_nvfp4(*entry["v"]).to(torch.float64).to(device)
            target = attention(qd.reshape(tokens, -1), kd.reshape(tokens, -1), v_ref)

            tm = toeplitz(w[0], tokens)
            probe = torch.randn(tokens, 4, dtype=torch.float64, device=device)
            gap = float((tm @ probe - banded(w[0], probe)).abs().max())
            print(f"  [L{layer:>2}/w{w_index}] banded-vs-Toeplitz max|gap| = {gap:.3e}", flush=True)

            step = (
                pv["scale_factor"].to(torch.float64)
                * pv["scale_lv2"].to(torch.float64)
                * pv["scale_lv3"].to(torch.float64)
                / 4.0
            ).expand_as(pv["mant"].to(torch.float64))
            sign = pv["sign"].to(torch.float64)
            # `code` is the INTEGER mantissa slot 0..7 (mant*4).  A legal +-1
            # step is one unit of this, not one unit of `mant` (which is
            # 0..1.75 in steps of 0.25).  Stepping `mant` by 1 and clamping to
            # [0, 7] saturated every touched code and inflated the measured
            # MSE by >100%; that was a rule bug, not a property of the metric.
            code = pv["mant"].to(torch.float64) * 4.0
            width = kv_heads * dim
            step_flat = (sign * step).reshape(tokens, width)
            movable = ((sign != 0) & (code > 0) & (code < 7)).reshape(tokens, width)
            base_code = code.reshape(tokens, width).clone()

            def residual(codes):
                vals = (codes * step_flat).reshape(tokens, kv_heads, dim)
                return vals - v_ref.reshape(tokens, kv_heads, dim)

            def kernel_objective(codes):
                resid = residual(codes)
                total = torch.zeros((), dtype=torch.float64, device=device)
                for g in range(kv_heads):
                    dg = resid[:, g, :]
                    for h in range(g * group, (g + 1) * group):
                        total = total + (banded(w[h], dg) ** 2).sum()
                return total

            def uniform_objective(codes):
                return residual(codes).sum(0).square().sum()

            def apply_W(wg, delta):
                """(T, C_g) -> (T, C_g): sum_offset wg[offset] * delta[t+offset]."""

                near, far = wg[:NBUCKET], wg[NFAR]
                pad = torch.zeros(KERNEL_RADIUS, delta.shape[1], dtype=delta.dtype, device=device)
                windows = torch.cat([pad, delta, pad], dim=0).unfold(0, NBUCKET, 1)
                total = delta.sum(0, keepdim=True)
                return torch.einsum("tcr,r->tc", windows, near) + far * (total - windows.sum(-1))

            def apply_WT(wg, sig):
                """The transpose action: (W^T sig)[i] = sum_t wg[i-t] sig[t]."""

                flipped = torch.cat([wg[:NBUCKET].flip(0), wg[NFAR:]]).contiguous()
                return apply_W(flipped, sig)

            def row_energy(wg, tokens):
                """(W^T W)_ii for each i -- the exact quadratic coefficient."""

                energy = torch.zeros(tokens, dtype=torch.float64, device=device)
                far_count = torch.zeros(tokens, dtype=torch.float64, device=device)
                for r in range(NBUCKET):
                    off = r - KERNEL_RADIUS
                    lo, hi = max(0, -off), min(tokens, tokens - off)
                    if hi <= lo:
                        continue
                    idx = torch.arange(lo, hi, device=device)
                    energy.index_add_(0, idx, wg[r].square().expand(hi - lo))
                    far_count.index_add_(0, idx, torch.ones(hi - lo, dtype=torch.float64, device=device))
                return energy + (tokens - far_count) * wg[NFAR].square()

            def greedy(kernel_mode):
                """One exact best single-element step per channel per sweep.

                J is a quadratic form in the codes and separable across channels,
                so for a change eps on element i the objective moves by exactly
                g_i*eps + eps^2 * Q_ii with g = 2 sum_h W_h^T (W_h d) and
                Q_ii = sum_h (W_h^T W_h)_ii.  Channels are independent, so the
                per-channel best steps can be applied simultaneously.
                """

                cur = base_code.clone()
                for _ in range(SWEEPS):
                    vals = (cur * step_flat).reshape(tokens, kv_heads, dim)
                    resid = vals - v_ref.reshape(tokens, kv_heads, dim)
                    grad = torch.zeros_like(resid)
                    qii = torch.zeros(tokens, dtype=torch.float64, device=device)
                    if kernel_mode:
                        for g in range(kv_heads):
                            dg = resid[:, g, :]
                            for h in range(g * group, (g + 1) * group):
                                wg = w[h]
                                sig = apply_W(wg, dg)
                                grad[:, g, :] += 2.0 * apply_WT(wg, sig)
                                qii += row_energy(wg, tokens)
                    # uniform mode: W = 1/T everywhere, so g = 2*sum_t delta / T^2
                    # and Q_ii = 1/T -- written out rather than routed through the
                    # banded helpers so the two rules stay independently checkable.
                    if not kernel_mode:
                        col = resid.sum(0, keepdim=True)
                        grad = 2.0 * col.expand_as(resid) / float(tokens)
                        qii = torch.full((tokens,), 1.0 / float(tokens), dtype=torch.float64, device=device)

                    n_chan = kv_heads * dim
                    best_gain = torch.full((n_chan,), float("inf"), dtype=torch.float64, device=device)
                    best_row = torch.zeros(n_chan, dtype=torch.long, device=device)
                    best_dir = torch.zeros(n_chan, dtype=torch.float64, device=device)
                    steps = step_flat                            # value change per +-1 code unit
                    flat_grad = grad.reshape(tokens, n_chan)
                    for direction in (1.0, -1.0):
                        eps = direction * steps
                        change = flat_grad * eps + (eps ** 2) * qii[:, None]
                        change = torch.where(
                            movable, change, torch.full_like(change, float("inf"))
                        )
                        gain, arg = change.min(dim=0)            # (n_chan,)
                        take = gain < best_gain
                        best_gain = torch.where(take, gain, best_gain)
                        best_row = torch.where(take, arg, best_row)
                        best_dir = torch.where(
                            take, torch.full_like(best_gain, direction), best_dir
                        )
                    if not bool((best_gain < 0).any()):
                        break
                    improved = best_gain < 0
                    cols = torch.arange(n_chan, device=device)
                    updated = (cur[best_row, cols] + best_dir).clamp(0.0, 7.0)
                    cur[best_row, cols] = torch.where(improved, updated, cur[best_row, cols])
                return cur

            ker_code = greedy(True)
            uni_code = greedy(False)

            def to_dense(codes):
                return (codes * step_flat).reshape(tokens, kv_heads * dim)

            # Independent check: reconstructing the PARENT codes through the
            # same path used to score the greedy must reproduce the parent's
            # decoded V exactly.  Without this, a wrong `code` scaling would
            # shift every measured MSE and still look like a plausible gain.
            v_parent_dense = dense(pv, shapes[2])
            rebuild_gap = float((to_dense(base_code) - v_parent_dense).abs().max())
            print(
                f"  [L{layer:>2}/w{w_index}] parent-code reconstruction max|gap| = {rebuild_gap:.3e}",
                flush=True,
            )

            def _validated(codes):
                """Decode through the real, VALIDATING decoder.

                `to_dense` is a hand-written step model that bypasses
                `reference_hif4.validate_hif4_params`, so a rule that produced an
                illegal mantissa would still be scored and could look like a gain.
                Routing through dequantize_hif4 makes that impossible.
                """
                out = {k: v.clone() for k, v in pv.items()}
                mm = (codes / 4.0).to(pv["mant"].dtype).reshape(pv["mant"].shape)
                out["mant"] = mm
                out["sign"] = torch.where(
                    mm == 0, torch.zeros_like(pv["sign"]), pv["sign"]
                ).to(pv["sign"].dtype)
                return v2.dequantize_hif4(v2._cpu_params(out), shapes[2]).to(torch.float64).to(device)

            def true_mse(codes):
                out = attention(qd.reshape(tokens, -1), kd.reshape(tokens, -1), _validated(codes))
                return float((out - target).square().mean())

            mse_par = true_mse(base_code)
            mse_ker = true_mse(ker_code)
            mse_uni = true_mse(uni_code)
            rows.append(
                {
                    "layer": layer,
                    "window": w_index,
                    "tokens": tokens,
                    "banded_gap": gap,
                    "parent_rebuild_gap": rebuild_gap,
                    "mse_parent": mse_par,
                    "mse_kernel": mse_ker,
                    "mse_uniform": mse_uni,
                    "kernel_vs_parent": mse_ker / mse_par - 1.0,
                    "uniform_vs_parent": mse_uni / mse_par - 1.0,
                    "moved_kernel": int((ker_code != base_code).sum()),
                    "moved_uniform": int((uni_code != base_code).sum()),
                }
            )
            print(
                f"  [L{layer:>2}/w{w_index} T={tokens:>4}] parent={mse_par:.6e}  "
                f"kernel {rows[-1]['kernel_vs_parent']:+.4%} (moved {rows[-1]['moved_kernel']})  "
                f"uniform {rows[-1]['uniform_vs_parent']:+.4%} (moved {rows[-1]['moved_uniform']})",
                flush=True,
            )

    if not rows:
        print("no windows under the token cap")
        return 1
    gaps = [r["banded_gap"] for r in rows]
    print()
    print(f"independent check: max banded-vs-Toeplitz gap over windows = {max(gaps):.3e}")
    kv = [r["kernel_vs_parent"] for r in rows]
    uv = [r["uniform_vs_parent"] for r in rows]
    print()
    print("relative change in TRUE attention output MSE vs the parent codes:")
    print(f"  KERNEL   mean={statistics.mean(kv):+.4%}  range=[{min(kv):+.4%}, {max(kv):+.4%}]  n={len(kv)}")
    print(f"  UNIFORM  mean={statistics.mean(uv):+.4%}  range=[{min(uv):+.4%}, {max(uv):+.4%}]  n={len(uv)}")
    print()
    print("NO THRESHOLD IS APPLIED (plan section 6.1).")
    (HERE / "vk3-rule.json").write_text(
        json.dumps(
            {"layers": list(LAYERS), "max_tokens": MAX_TOKENS, "sweeps": SWEEPS, "rows": rows},
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"wrote {HERE / 'vk3-rule.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
