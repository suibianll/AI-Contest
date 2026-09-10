"""A-JC1's line search, redone on real data: setting artefact, or the mechanism?

The synthetic contract found no scale at which the analytic direction improves the
hard encoding -- but that ran on 8 tokens, 2 Q heads, head_dim 64, a random draw
and a standard-HiF4 parent.  Anything learned from a toy setting has to be
re-tested before it is believed.

Same search, on the real panel path:

  * real layer 22 windows, real geometry (16 Q / 4 KV / head_dim 256);
  * the parent encoding from the *root's own* dynamic Q/K APIs, including the
    learned rotation and K center, so the encoded operand is the true
    post-transform one;
  * V frozen at the parent decode, exactly as the contract fixes it;
  * target = the NVFP4 reference decode's attention, matching the evaluator;
  * the contract's own solve -- jacobian, damped normal equations -- unchanged.

Only the rule shape is carried over from the contract: per-element features, 8
parameters, the same rounded hard encode.

Reading is deliberately narrow: this says whether *this direction* has a usable
scale on *these real cases*, nothing about the panel, and no verdict on the card.

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

SCALES = (0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 4.0)
MAX_TOKENS = 128  # bound the jacobian's memory; small windows only


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def main() -> int:
    torch.set_num_threads(2)
    torch.set_grad_enabled(False)
    solution = load("ajc1_real_solution", ROOT / "solution.py")
    smoke = load("ajc1_real_smoke", HERE / "contract_smoke.py")

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
    layer = 22

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

    def attention(q, k, v):
        return v2._attention(*(t[None] for t in (q, k, v)), q_heads, kv_heads, head_dim)

    def post_transform(reference, state, heads):
        """The operand the encoder actually saw, per the contract's pipeline."""

        dense = reference
        rotation = state.get("learned_rotation")
        if rotation is not None:
            dense = solution._a2_apply_group_rotation(
                dense, int(heads), rotation.to(torch.float32)
            )
        center = state.get("learned_center")
        if center is not None:
            lead = dense.shape[:-1]
            hd = dense.shape[-1] // int(heads)
            dense = (
                dense.reshape(*lead, int(heads), hd)
                + center.to(torch.float32).reshape(*([1] * len(lead)), int(heads), hd)
            ).reshape(dense.shape)
        return dense

    rows = []
    for window in range(len(pack["test_qkv"])):
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

        fq, bq = smoke.features(post_transform(refs[0], states["q_state"], q_heads), pq)
        fk, bk = smoke.features(post_transform(refs[1], states["k_state"], kv_heads), pk)
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
            return int((aq["mant"] != pq["mant"]).sum()), int((ak["mant"] != pk["mant"]).sum())

        parent = hard_mse(zero)
        scan = []
        for scale in SCALES:
            t = theta * scale
            cq, ck = changed(t)
            scan.append(
                {
                    "scale": scale,
                    "hard_mse": hard_mse(t),
                    "changed_q": cq,
                    "changed_k": ck,
                }
            )
        best = min(scan, key=lambda r: r["hard_mse"])
        rows.append({"window": window, "tokens": int(pairs[0][0].shape[0]),
                     "parent_hard_mse": parent, "scan": scan})
        print(
            f"[layer{layer}/win{window} tokens={int(pairs[0][0].shape[0])}] parent {parent:.8f} | "
            f"best scale {best['scale']:.2f} -> {best['hard_mse']:.8f} "
            f"({100*(best['hard_mse']-parent)/parent:+.3f}%) q/k {best['changed_q']}/{best['changed_k']} | "
            f"zero-change scales {[r['scale'] for r in scan if r['changed_q']==0 and r['changed_k']==0]}",
            flush=True,
        )

    improved = sum(
        1 for r in rows if min(s["hard_mse"] for s in r["scan"]) < r["parent_hard_mse"]
    )
    print()
    print(f"windows={len(rows)} | some scale beats the parent in {improved}")

    (HERE / "ajc1-real-line-search.json").write_text(
        json.dumps(
            {
                "scope": "real layer 22 windows with the root's own parent encoding; CPU only",
                "layer": layer,
                "max_tokens": MAX_TOKENS,
                "windows": len(rows),
                "windows_improved": improved,
                "rows": rows,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"wrote {HERE / 'ajc1-real-line-search.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
