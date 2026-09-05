"""S1 target-unit structure autopsy for the fc/proj & Q/K research plan.

Offline, read-only.  Reuses the cached v186 calibration artifacts and the
validated coordinate mirrors to characterise, for the units where official
gain was proven to live (fc_gate/fc_up/proj weights; Attention Q/K), whether
the remaining quantization error is clip-limited or round-limited.

For each encoding we compute per-element  r = |x| / (scale_factor*lv2*lv3),
the mantissa-grid coordinate in units of the finest 0.25 step, and report:
clip_frac (r>=1.75), high_frac (r>=0.75), low_frac (0<r<0.5), zero_frac,
r p50/p90/p99/max and mean_r2.  No version is produced and no timing is
recorded; this is research evidence only.

Usage: python workbench/s1_target_autopsy.py [--device cuda]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "evaluator"))

import official_eval as v2  # noqa: E402
import proxy_v3_eval as v3  # noqa: E402
import coordinate_diagnostics as diag  # noqa: E402

SOLUTION = ROOT / "solution.py"
CACHE = ROOT / "artifacts" / "official_eval" / "cache" / "qwen2.5-0.5b-proxy-v2.pt"
OUT = ROOT / "artifacts" / "proxy_v3" / "targeted-autopsy-20260905" / "run-001"
ROLES = ("fc_gate", "fc_up", "proj")


def r_stats(x: torch.Tensor, params: Any) -> dict[str, float]:
    """Per-element mantissa-grid coordinate stats for a continuous tensor + params."""
    sf = params["scale_factor"]
    lv2 = params["scale_lv2"]
    lv3 = params["scale_lv3"]
    denom = sf * lv2 * lv3  # (..., blocks, 8, 2, 1)
    xb = x.to(torch.float32)
    channels = int(xb.shape[-1])
    blocks = channels // 64
    xb = xb.reshape(*xb.shape[:-1], blocks, 8, 2, 4)
    r = xb.abs() / denom.clamp_min(1.0e-30)
    r = r.reshape(-1)
    finite = torch.isfinite(r)
    r = r[finite]
    n = r.numel()
    if n == 0:
        return {"n": 0}
    q = torch.quantile(r, torch.tensor([0.5, 0.9, 0.99], device=r.device))
    return {
        "n": int(n),
        "clip_frac": float((r >= 1.75).float().mean()),
        "high_frac": float((r >= 0.75).float().mean()),
        "low_frac": float(((r > 0.0) & (r < 0.5)).float().mean()),
        "zero_frac": float((r == 0.0).float().mean()),
        "r_p50": float(q[0]),
        "r_p90": float(q[1]),
        "r_p99": float(q[2]),
        "r_max": float(r.max()),
        "mean_r2": float((r * r).mean()),
    }


def collect_weights(sol: Any, device: torch.device) -> list[dict[str, Any]]:
    raw = v2.load_pack(CACHE)
    out: list[dict[str, Any]] = []
    for shard in range(6):
        pack = v3.prepare_shard(raw, shard, "both", ood=False)
        identity = v3._calibration_identity(SOLUTION, pack, device)
        cache_path = v3.default_calibration_cache_path(identity)
        weight_states, attention_states = v3.load_calibration_artifact(
            cache_path, identity, pack
        )
        keys = sorted(weight_states)
        for key in keys:
            layer, role = key
            if role not in ROLES:
                continue
            state, params = weight_states[key]
            w_pair = pack.weights[layer][role]
            w_raw = v2.dequantize_nvfp4(*w_pair).to(torch.float32)
            w_t = diag.linear_weight_continuous(w_raw, state, sol)
            stats = r_stats(w_t, params)
            stats.update({"layer": layer, "role": role, "shard": shard, "rows": int(w_raw.shape[0]), "cols": int(w_raw.shape[1])})
            out.append(stats)
        v3.cleanup_solution_modules()
    return out


def collect_attention_qk(sol: Any, device: torch.device) -> list[dict[str, Any]]:
    raw = v2.load_pack(CACHE)
    out: list[dict[str, Any]] = []
    for shard in range(6):
        pack = v3.prepare_shard(raw, shard, "both", ood=False)
        identity = v3._calibration_identity(SOLUTION, pack, device)
        cache_path = v3.default_calibration_cache_path(identity)
        _, attention_states = v3.load_calibration_artifact(cache_path, identity, pack)
        for case in pack.attention_cases:
            states = attention_states[case.layer]
            pairs = pack.test_qkv[case.test_window][case.layer]
            q_pair, k_pair, _ = (v2._move_pair(pair, device) for pair in pairs)
            q_raw = v2.dequantize_nvfp4(*q_pair).to(torch.float32)
            k_raw = v2.dequantize_nvfp4(*k_pair).to(torch.float32)
            q_t = sol._attention_state_transform_dense(
                q_raw, states["q_state"], pack.q_heads, pack.head_dim, is_k=False
            ).to(torch.float32)
            k_t = sol._attention_state_transform_dense(
                k_raw, states["k_state"], pack.kv_heads, pack.head_dim, is_k=True
            ).to(torch.float32)
            q_params = sol.hif4_dynamic_quantize_q(
                q_pair[0], q_pair[1], pack.q_heads, pack.head_dim, states["q_state"]
            )
            k_params = sol.hif4_dynamic_quantize_k(
                k_pair[0], k_pair[1], pack.kv_heads, pack.head_dim, states["k_state"]
            )
            qs = r_stats(q_t, q_params)
            ks = r_stats(k_t, k_params)
            qs.update({"layer": case.layer, "operand": "q", "test_length": int(q_raw.shape[0]), "split": pack.test_windows[case.test_window].split})
            ks.update({"layer": case.layer, "operand": "k", "test_length": int(k_raw.shape[0]), "split": pack.test_windows[case.test_window].split})
            out.extend([qs, ks])
        v3.cleanup_solution_modules()
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    device = torch.device(args.device)
    OUT.mkdir(parents=True, exist_ok=True)
    sol = v2.load_solution(SOLUTION)
    start = time.perf_counter()
    weight_rows = collect_weights(sol, device)
    attention_rows = collect_attention_qk(sol, device)
    payload = {
        "protocol": "targeted-unit-autopsy-v1",
        "plan": "2026-09-05-targeted-fcproj-qk-mechanism-research-plan",
        "source_sha256": v2.sha256_file(SOLUTION),
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "wall_seconds": time.perf_counter() - start,
        "roles": list(ROLES),
        "weight_rows": weight_rows,
        "attention_rows": attention_rows,
    }
    (OUT / "autopsy.json").write_text(json.dumps(payload, indent=1), encoding="utf-8")
    md = ["# S1 fc/proj + Q/K 结构解剖（v186）", "", f"- {payload['created_at']}",
          "", "## fc/proj 权重逐 (layer, role)", "",
          "| layer | role | clip% | high% | low% | zero% | p50 | p90 | p99 | max |",
          "|---|---|---|---|---|---|---|---|---|"]
    for row in weight_rows:
        md.append(f"| {row['layer']} | {row['role']} | {row['clip_frac']*100:.2f} | {row['high_frac']*100:.1f} | {row['low_frac']*100:.1f} | {row['zero_frac']*100:.1f} | {row['r_p50']:.3f} | {row['r_p90']:.3f} | {row['r_p99']:.3f} | {row['r_max']:.3f} |")
    md.append("")
    md.append("## Attention Q/K 逐 (layer, operand)")
    md.append("| layer | op | clip% | high% | low% | zero% | p50 | p90 | p99 | max |")
    md.append("|---|---|---|---|---|---|---|---|---|---|")
    for row in attention_rows:
        md.append(f"| {row['layer']} | {row['operand']} | {row['clip_frac']*100:.2f} | {row['high_frac']*100:.1f} | {row['low_frac']*100:.1f} | {row['zero_frac']*100:.1f} | {row['r_p50']:.3f} | {row['r_p90']:.3f} | {row['r_p99']:.3f} | {row['r_max']:.3f} |")
    (OUT / "autopsy.md").write_text("\n".join(md), encoding="utf-8")
    print(f"wrote {OUT / 'autopsy.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
