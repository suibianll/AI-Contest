"""A-JC1's three outstanding checks, on all six real layers.

`ajc1_real_linesearch.py` ran the contract's own solve on **layer 22 only**, with
tokens capped at 128, and reported whether some scale of the analytic direction
beats the parent's hard encoding.  That is one layer out of six, and the plan
(`proposals/2026-09-10-attention-080-algorithm-research-next.md` §3) names three
things that were never checked at all:

  1. the parameter direction itself -- `g_theta = J_theta^T r` -- on real data;
  2. the actual hard-code change the compiled rule produces;
  3. reachability: whether the rule can move codes at all, per layer.

This is not a feasibility measure.  It reports those three quantities and nothing
about the panel, the 0.80 milestone, or the card's official value.  The scored
question -- does the panel improve -- is decided by the six-shard run, not here.

Everything is the contract's: per-element features, 8 parameters, the same
rounded hard encode, the same damped normal-equation solve, V frozen at the
parent decode, target = the NVFP4 reference decode's attention.

CPU only, no shard, no candidate.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]

LAYERS = (0, 1, 5, 8, 15, 22)
SCALES = (0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 4.0)
MAX_TOKENS = 96  # the jacobian is (out*head_dim, 8); keep it small on CPU
MAX_WINDOWS = 4


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def post_transform(dense, state, heads, solution):
    """The operand the encoder actually sees, per the contract's pipeline."""

    rotation = state.get("learned_rotation")
    if rotation is not None:
        dense = solution._a2_apply_group_rotation(dense, int(heads), rotation.to(torch.float32))
    center = state.get("learned_center")
    if center is not None:
        lead = dense.shape[:-1]
        hd = dense.shape[-1] // int(heads)
        dense = (
            dense.reshape(*lead, int(heads), hd)
            + center.to(torch.float32).reshape(*([1] * len(lead)), int(heads), hd)
        ).reshape(dense.shape)
    return dense


def main() -> int:
    torch.set_num_threads(4)
    torch.set_grad_enabled(False)
    solution = load("six_solution", ROOT / "solution.py")
    smoke = load("six_smoke", HERE / "contract_smoke.py")

    sys.path.insert(0, str(ROOT / "evaluator"))
    import official_eval as v2  # noqa: PLC0415

    pack = torch.load(
        ROOT / "artifacts/official_eval/cache/qwen3.5-4b-proxy-v2.pt",
        map_location="cpu", mmap=True, weights_only=False,
    )
    q_heads = int(pack["q_heads"])
    kv_heads = int(pack["kv_heads"])
    head_dim = int(pack["head_dim"])
    device = torch.device("cpu")

    def attention(q, k, v):
        return v2._attention(*(t[None] for t in (q, k, v)), q_heads, kv_heads, head_dim)

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
        states = solution.hif4_calibration_attention(windows, q_heads, kv_heads, head_dim)

        for window in range(min(MAX_WINDOWS, len(pack["test_qkv"]))):
            entry = pack["test_qkv"][window][layer]
            if entry is None:
                continue
            pairs = [v2._pair(entry[i].to(torch.float32)) for i in range(3)]
            if int(pairs[0][0].shape[0]) > MAX_TOKENS:
                continue
            refs = [v2.dequantize_nvfp4(*p).to(torch.float32) for p in pairs]

            pq = solution.hif4_dynamic_quantize_q(*pairs[0], q_heads, head_dim, states["q_state"])
            pk = solution.hif4_dynamic_quantize_k(*pairs[1], kv_heads, head_dim, states["k_state"])
            pv = solution.hif4_dynamic_quantize_v(*pairs[2], kv_heads, head_dim, states["v_state"])
            q0 = v2.dequantize_hif4(v2._cpu_params(pq), refs[0].shape).to(torch.float64)
            k0 = v2.dequantize_hif4(v2._cpu_params(pk), refs[1].shape).to(torch.float64)
            v0 = v2.dequantize_hif4(v2._cpu_params(pv), refs[2].shape).to(torch.float64)

            fq, bq = smoke.features(post_transform(refs[0], states["q_state"], q_heads, solution), pq)
            fk, bk = smoke.features(post_transform(refs[1], states["k_state"], kv_heads, solution), pk)
            bq = bq.to(torch.float64)
            bk = bk.to(torch.float64)

            target = attention(refs[0], refs[1], refs[2]).double()
            zero = torch.zeros(8, dtype=torch.float64)

            def relaxed(t):
                return attention(q0 + bq @ t[:4], k0 + bk @ t[4:], v0)

            residual = (relaxed(zero) - target).flatten()
            jac = torch.autograd.functional.jacobian(relaxed, zero).reshape(-1, 8)
            h = jac.T @ jac / residual.numel()
            g = jac.T @ residual / residual.numel()
            ridge = max(1e-4 * float(h.trace()) / 8, 1e-12)
            theta = torch.linalg.solve(h + ridge * torch.eye(8, dtype=h.dtype), -g)

            # Check 1: the parameter direction itself, against the raw residual.
            #
            # NOTE: cos(g, theta) is negative *by construction* -- theta solves
            # (H + ridge I) theta = -g, so g.theta = -g^T H^-1 g < 0 whenever H is
            # positive definite.  It is therefore not evidence of anything and
            # must not be quoted as a finding; it is printed only as a sanity
            # check that the solve returned the descent direction rather than
            # its opposite.  What check 1 actually reports is that |g| is
            # non-zero and that the relaxed objective falls at theta.
            g_norm = float(g.norm())
            t_norm = float(theta.norm())
            cos = (
                float((g @ theta) / (g.norm() * theta.norm()))
                if g_norm > 0 and t_norm > 0
                else float("nan")
            )
            # The cosine alone would be signed; the useful reading is whether the
            # step lowers the *relaxed* objective, which the scan below shows on
            # the hard encode.
            relaxed_0 = float((relaxed(zero) - target).square().mean())
            relaxed_1 = float((relaxed(theta) - target).square().mean())

            def encode(t):
                return smoke.hard(pq, fq, t[:4]), smoke.hard(pk, fk, t[4:])

            def hard_mse(t):
                aq, ak = encode(t)
                out = attention(
                    v2.dequantize_hif4(v2._cpu_params(aq), refs[0].shape).double(),
                    v2.dequantize_hif4(v2._cpu_params(ak), refs[1].shape).double(),
                    v0,
                )
                return float((out - target).square().mean())

            def changed(t):
                aq, ak = encode(t)
                return int((aq["mant"] != pq["mant"]).sum()) + int((ak["mant"] != pk["mant"]).sum())

            parent = hard_mse(zero)
            scan = [
                {"scale": s, "hard_mse": hard_mse(theta * s), "changed": changed(theta * s)}
                for s in SCALES
            ]
            best = min(scan, key=lambda r: r["hard_mse"])
            rows.append(
                {
                    "layer": layer,
                    "window": window,
                    "tokens": int(pairs[0][0].shape[0]),
                    "grad_norm": g_norm,
                    "theta_norm": t_norm,
                    "theta_grad_cos": cos,
                    "relaxed_parent": relaxed_0,
                    "relaxed_theta": relaxed_1,
                    "parent_hard_mse": parent,
                    "scan": scan,
                    "best_scale": best["scale"],
                    "best_hard_mse": best["hard_mse"],
                    "best_changed": best["changed"],
                }
            )
            print(
                f"[L{layer:>2}/w{window} n={rows[-1]['tokens']:>3}] "
                f"|g|={g_norm:.3e} |t|={t_norm:.3e} cos(g,t)={cos:+.3f}[-by-construction] "
                f"relaxed {relaxed_0:.3e}->{relaxed_1:.3e} | "
                f"hard {parent:.8f} best@{best['scale']:.2f}={best['hard_mse']:.8f} "
                f"({100*(best['hard_mse']-parent)/parent:+.3f}%) changed={best['changed']} "
                f"zero-change={[r['scale'] for r in scan if r['changed'] == 0]}",
                flush=True,
            )

    print()
    print(f"cases={len(rows)}")
    per_layer = {}
    for r in rows:
        per_layer.setdefault(r["layer"], []).append(r)
    for layer, sub in sorted(per_layer.items()):
        improved = sum(1 for r in sub if r["best_hard_mse"] < r["parent_hard_mse"])
        reachable = sum(1 for r in sub if r["best_changed"] > 0)
        print(
            f"  layer {layer:>2}: cases={len(sub)}  hard-improved={improved}  "
            f"codes-moved={reachable}  mean-cos={sum(r['theta_grad_cos'] for r in sub)/len(sub):+.3f}"
        )
    improved = sum(1 for r in rows if r["best_hard_mse"] < r["parent_hard_mse"])
    reachable = sum(1 for r in rows if r["best_changed"] > 0)
    print(f"TOTAL hard-improved={improved}/{len(rows)}  codes-moved={reachable}/{len(rows)}")

    (HERE / "ajc1-six-layer.json").write_text(
        json.dumps(
            {
                "scope": "six real attention layers, root's own parent encoding; CPU only",
                "layers": list(LAYERS),
                "max_tokens": MAX_TOKENS,
                "max_windows": MAX_WINDOWS,
                "scales": list(SCALES),
                "rows": rows,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"wrote {HERE / 'ajc1-six-layer.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
