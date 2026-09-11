"""Where is the Attention error, in the SCORING metric?

The panel runs with `error_source_decomposition: false`, so no panel artifact has
ever carried the Q/K/V split, and the two loose numbers that were once quoted for
it turned out to belong to other quantities entirely (defect #36).  This measures
it directly.

It re-runs the proxy-v3 scoring loop for the attention cases -- copied from
`evaluator/proxy_v3_eval.py:421-470` rather than re-derived, which is the rule
from defect #19 -- and adds the two arms the evaluator's own decomposition uses
(`official_eval.py:1693-1728`):

    reference  attn(q_ref,      k_ref,      v_ref)
    standard   attn(q_std,      k_std,      v_std)      <- the denominator
    player     attn(q_hat,      k_hat,      v_hat)      <- the numerator
    v_only     attn(q_std,      k_std,      v_hat)      gain_v_only  = (e_std - e_v_only)/e_std
    qk_only    attn(q_hat,      k_hat,      v_std)      gain_qk_only = (e_std - e_qk_only)/e_std

`gain_q_only` and `gain_k_only` use the same idea one operand at a time.  All five
lands are reported as gains, so they are directly comparable to the panel's own
`gain`, and the sanity arm (reference vs reference) must be exactly zero.

    AD1_DEVICE=cuda .venv/Scripts/python.exe ad1_decomp.py
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


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def main() -> int:
    torch.set_num_threads(4)
    torch.set_grad_enabled(False)
    solution = load("ad1_sol", ROOT / "solution.py")
    sys.path.insert(0, str(ROOT / "evaluator"))
    import official_eval as v2          # noqa: PLC0415
    import proxy_v3_eval as v3          # noqa: PLC0415

    dev = torch.device(os.environ.get("AD1_DEVICE", "cuda"))
    pack_path = ROOT / "artifacts/official_eval/cache/qwen3.5-4b-proxy-v2.pt"
    rawpack = v2.load_pack(pack_path)
    print("loading solution calibration ...", flush=True)

    # Calibrate attention states exactly as the panel does.
    states_by_layer = {}
    for shard in range(6):
        pack = v3.prepare_shard(rawpack, shard, "attention")
        for layer in pack.metadata["attention_state_layers"]:
            if layer in states_by_layer:
                continue
            calibration = [
                v2._move_qkv(pack.calibration_qkv[sample][layer], dev)
                for sample in range(len(pack.calibration_windows))
            ]
            states_by_layer[layer] = solution.hif4_calibration_attention(
                calibration, pack.q_heads, pack.kv_heads, pack.head_dim
            )
    print(f"calibrated {len(states_by_layer)} attention layers", flush=True)

    rows = []
    for shard in range(6):
        pack = v3.prepare_shard(rawpack, shard, "attention")
        for case in pack.attention_cases:
            states = states_by_layer[case.layer]
            pairs = pack.test_qkv[case.test_window][case.layer]
            q_pair, k_pair, value_pair = (v2._move_pair(p, dev) for p in pairs)
            params = {}
            for api_name, fn, pair, heads, sname in (
                ("q", solution.hif4_dynamic_quantize_q, q_pair, pack.q_heads, "q_state"),
                ("k", solution.hif4_dynamic_quantize_k, k_pair, pack.kv_heads, "k_state"),
                ("v", solution.hif4_dynamic_quantize_v, value_pair, pack.kv_heads, "v_state"),
            ):
                params[api_name] = fn(pair[0], pair[1], heads, pack.head_dim, states[sname])
            refs = [v2.dequantize_nvfp4(*p).to(torch.float32) for p in pairs]
            stds = [v2.decode_standard_hif4(v2.encode_standard_hif4(x)).to(torch.float32) for x in refs]
            hats = [
                v2.dequantize_hif4(v2._cpu_params(params[n]), r.shape).to(torch.float32)
                for n, r in zip(("q", "k", "v"), refs)
            ]
            q_hat, k_hat, v_hat = hats
            q_std, k_std, v_std = stds
            q_ref, k_ref, v_ref = refs
            qh, kvh, hd = pack.q_heads, pack.kv_heads, pack.head_dim

            def attn(q, k, v):
                return v2._attention(q[None].to(dev), k[None].to(dev), v[None].to(dev), qh, kvh, hd)

            ref_o = attn(q_ref, k_ref, v_ref)
            std_o = attn(q_std, k_std, v_std)
            ply_o = attn(q_hat, k_hat, v_hat)
            arms = {
                "v_only": attn(q_std, k_std, v_hat),
                "qk_only": attn(q_hat, k_hat, v_std),
                "q_only": attn(q_hat, k_std, v_std),
                "k_only": attn(q_std, k_hat, v_std),
                "identity_check": attn(q_ref, k_ref, v_ref),
            }
            e = lambda a: float((a - ref_o).square().mean())      # noqa: E731
            e_std = e(std_o)
            row = {
                "shard": shard, "case_id": case.case_id, "layer": case.layer,
                "length": int(refs[0].shape[0]),
                "gain": (e_std - e(ply_o)) / e_std,
                "mse_standard": e_std, "mse_player": e(ply_o),
                "identity_check_mse": e(arms["identity_check"]),
            }
            for name in ("v_only", "qk_only", "q_only", "k_only"):
                row["gain_" + name] = (e_std - e(arms[name])) / e_std
            rows.append(row)
        print(f"  shard {shard}: {len(rows)} cases so far", flush=True)

    out = HERE / "ad1-decomposition.json"
    out.write_text(json.dumps(rows, indent=1), encoding="utf-8")

    n = len(rows)
    print(f"\n  cases: {n}")
    print(f"  sanity: max identity-check mse = {max(r['identity_check_mse'] for r in rows):.3e} (must be ~0)")
    print(f"\n  {'quantity':<16}{'mean':>10}{'min':>10}{'max':>10}")
    keys = ["gain", "gain_q_only", "gain_k_only", "gain_v_only", "gain_qk_only"]
    for k in keys:
        v = [r[k] for r in rows]
        print(f"  {k:<16}{statistics.mean(v):>+10.4f}{min(v):>+10.4f}{max(v):>+10.4f}")
    print(f"\n  denominator: mean mse_standard = {statistics.mean(r['mse_standard'] for r in rows):.4e}")
    print(f"  numerator  : mean mse_player   = {statistics.mean(r['mse_player'] for r in rows):.4e}")
    print()
    print("  gain_v_only  = deploying V alone (Q,K standard)")
    print("  gain_qk_only = deploying Q,K alone (V standard)")
    print("  These are the evaluator's own definitions (official_eval.py:1719-1728).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
