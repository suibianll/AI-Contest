"""Could the C76.4 rotation search select per KV group instead of per layer?

The attention output is (tokens, q_heads * head_dim) and the score is an elementwise
MSE, so the error decomposes EXACTLY across heads: every element belongs to exactly
one head.  Grouping heads by their KV group therefore gives an exact per-group
error, and both the gate and the rotation argmin could in principle be taken per
group rather than once per layer.

That only buys anything if the BEST block differs between groups.  This measures
it: it wraps `_attention_deployed_mse` so every evaluated rotation candidate also
reports its per-head causal MSE, and tags the candidate with the `rotation_block`
the state carries.  For each layer it then asks whether the argmin block is the
same for all KV groups.

    APG1_DEVICE=cuda .venv/Scripts/python.exe apg1_probe.py
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
from collections import defaultdict
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
    root = load("apg1", ROOT / "solution.py")
    sys.path.insert(0, str(ROOT / "evaluator"))
    import official_eval as v2          # noqa: PLC0415

    dev = torch.device(os.environ.get("APG1_DEVICE", "cuda"))
    pack = torch.load(
        ROOT / "artifacts/official_eval/cache/qwen3.5-4b-proxy-v2.pt",
        map_location="cpu", mmap=True, weights_only=False,
    )
    qh = int(pack["q_heads"]); kvh = int(pack["kv_heads"]); hd = int(pack["head_dim"])
    group = qh // kvh
    splits = len(pack["calibration_qkv"])
    layers = [l for l in range(len(pack["calibration_qkv"][0]))
              if all(pack["calibration_qkv"][s][l] is not None for s in range(splits))]
    print(f"device={dev} q_heads={qh} kv_heads={kvh} head_dim={hd} groups={kvh}", flush=True)

    orig = root._attention_deployed_mse
    records = []

    def patched(q_pairs, k_pairs, v_hats, refs, q_state, k_state, qn, kn, hdim):
        causal, safety = orig(q_pairs, k_pairs, v_hats, refs, q_state, k_state, qn, kn, hdim)
        # per-head causal MSE on the same windows, computed the same way the
        # parent computes the scalar: mean over (tokens, head_dim) per head.
        per_head = []
        for (qq, qs), (kq, ks), v_hat, (ref_c, _ref_n) in zip(q_pairs, k_pairs, v_hats, refs):
            q_hat = root._dequantize_hif4(
                root.hif4_dynamic_quantize_q(qq, qs, qn, hdim, q_state)
            ).to(torch.float32)
            k_hat = root._dequantize_hif4(
                root.hif4_dynamic_quantize_k(kq, ks, kn, hdim, k_state)
            ).to(torch.float32)
            out_c = v2._attention(q_hat[None], k_hat[None], v_hat[None], qn, kn, hdim)[0]
            d = (out_c - ref_c).square().reshape(out_c.shape[0], qn, hdim).mean(dim=(0, 2))
            per_head.append(d)
        per_head = torch.stack(per_head).mean(dim=0)          # (q_heads,)
        per_group = per_head.reshape(kvh, group).mean(dim=1)  # (kv_heads,)
        records.append({
            "block": q_state.get("rotation_block"),
            "per_head_causal": [float(x) for x in per_head],
            "per_group_causal": [float(x) for x in per_group],
            "scalar_causal_mean": float(sum(causal) / len(causal)),
        })
        return causal, safety

    root._attention_deployed_mse = patched

    out = {}
    for layer in layers:
        records.clear()
        calibration = [
            {r: tuple(t.to(dev) for t in v2._pair(pack["calibration_qkv"][s][layer][i]))
             for i, r in enumerate(("q", "k", "v"))}
            for s in range(splits)
        ]
        root.hif4_calibration_attention(calibration, qh, kvh, hd)
        by_block = defaultdict(list)
        for rec in records:
            by_block[rec["block"]].append(rec)
        # keep, for each block, the candidate with the lowest scalar (the one the
        # search would prefer at that block)
        best = {}
        for block, recs in by_block.items():
            r = min(recs, key=lambda x: x["scalar_causal_mean"])
            best[block] = r
        out[layer] = {str(b): {"per_group": r["per_group_causal"],
                               "scalar": r["scalar_causal_mean"]} for b, r in best.items()}
        blocks = sorted(b for b in best if b is not None)
        argmins = []
        for g in range(kvh):
            vals = {b: best[b]["per_group_causal"][g] for b in blocks}
            argmins.append(min(vals, key=vals.get))
        same = len(set(argmins)) == 1
        print(f"layer {layer:>3}: blocks={blocks}  per-group argmin={argmins}  "
              f"{'SAME' if same else '*** DIFFERS ***'}", flush=True)
        for b in blocks:
            print(f"          block {b:>3}: per-group mse {[round(x, 6) for x in best[b]['per_group_causal']]}"
                  f"  scalar {best[b]['scalar_causal_mean']:.6f}", flush=True)

    (HERE / "apg1-pergroup.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print()
    print("If the per-group argmin block differs between groups, per-group selection has room.")
    print("If it is the same everywhere, per-group selection is a no-op and this direction closes.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
