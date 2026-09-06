"""L-branch frozen-side control check (v162-independent / Linear).

Verifies, on REAL captured Q/K/V inputs and real call order, that the four
Attention APIs of the L candidate are bit-identical to the v162 standard
file: five fields, states, and outputs.  Also probes linear-side mechanism
reachability (non-empty weight calibration state) on a few (layer, role)
pairs.

Usage (run under the branch GPU lock):
  .venv/Scripts/python.exe workbench/v162_linear/control_check.py \
      --candidate solutions/20260903_v163_v160-linear_standard-attn_scoreNA_timeNA/solution.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "evaluator"))

import official_eval as v2  # noqa: E402
import proxy_v3_eval as v3  # noqa: E402

CACHE = ROOT / "artifacts" / "official_eval" / "cache" / "qwen2.5-0.5b-proxy-v2.pt"
V162 = ROOT / "solutions" / "20260903_v162_standard-baseline-both_scoreNA_timeNA" / "solution.py"

BAD: list[str] = []


def _load(path: Path) -> Any:
    return v2.load_solution(path.resolve())


def _same(a: Any, b: Any, label: str) -> None:
    if isinstance(a, dict):
        if set(a) != set(b):
            BAD.append(f"{label}: key mismatch {sorted(map(str, a))} vs {sorted(map(str, b))}")
            return
        for key in a:
            _same(a[key], b[key], f"{label}.{key}")
        return
    if torch.is_tensor(a):
        if a.shape != b.shape:
            BAD.append(f"{label}: shape {tuple(a.shape)} vs {tuple(b.shape)}")
            return
        if not torch.equal(a.detach().cpu(), b.detach().cpu()):
            diff = float((a.float() - b.float()).abs().max())
            BAD.append(f"{label}: max abs diff {diff}")
        return
    if a != b:
        BAD.append(f"{label}: {a!r} != {b!r}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    device = torch.device(args.device)

    cand = _load(Path(args.candidate))
    ref = _load(V162)

    raw = v2.load_pack(CACHE)
    pack = v3.prepare_shard(raw, args.shard, "attention", ood=False)

    # Real calibration call order: candidate first, then reference, on the
    # same calibration qkv lists; both states must be empty dicts and equal.
    _, cand_attn_states, _, _ = v3._calibrate(cand, pack, device)
    _, ref_attn_states, _, _ = v3._calibrate(ref, pack, device)

    n_checked = 0
    first_pass: dict[int, tuple[Any, Any, Any]] = {}
    for case in pack.attention_cases:
        c_states = cand_attn_states[case.layer]
        r_states = ref_attn_states[case.layer]
        _same(c_states, r_states, f"L{case.layer} calib-states")
        pairs = pack.test_qkv[case.test_window][case.layer]
        q_pair, k_pair, v_pair = (v2._move_pair(p, device) for p in pairs)
        cq = cand.hif4_dynamic_quantize_q(q_pair[0], q_pair[1], pack.q_heads, pack.head_dim, c_states["q_state"])
        ck = cand.hif4_dynamic_quantize_k(k_pair[0], k_pair[1], pack.kv_heads, pack.head_dim, c_states["k_state"])
        cv = cand.hif4_dynamic_quantize_v(v_pair[0], v_pair[1], pack.kv_heads, pack.head_dim, c_states["v_state"])
        rq = ref.hif4_dynamic_quantize_q(q_pair[0], q_pair[1], pack.q_heads, pack.head_dim, r_states["q_state"])
        rk = ref.hif4_dynamic_quantize_k(k_pair[0], k_pair[1], pack.kv_heads, pack.head_dim, r_states["k_state"])
        rv = ref.hif4_dynamic_quantize_v(v_pair[0], v_pair[1], pack.kv_heads, pack.head_dim, r_states["v_state"])
        _same(cq, rq, f"L{case.layer} w{case.test_window} q")
        _same(ck, rk, f"L{case.layer} w{case.test_window} k")
        _same(cv, rv, f"L{case.layer} w{case.test_window} v")
        first_pass[case.case_id] = (cq, ck, cv)
        n_checked += 1
    print(f"attention control: {n_checked} cases compared (shard {args.shard})")

    # Reverse-order replay on the same cases: outputs must match the first
    # pass exactly (no global mutable state pollution on the frozen side).
    for case in reversed(pack.attention_cases):
        c_states = cand_attn_states[case.layer]
        pairs = pack.test_qkv[case.test_window][case.layer]
        q_pair, k_pair, v_pair = (v2._move_pair(p, device) for p in pairs)
        cq2 = cand.hif4_dynamic_quantize_q(q_pair[0], q_pair[1], pack.q_heads, pack.head_dim, c_states["q_state"])
        ck2 = cand.hif4_dynamic_quantize_k(k_pair[0], k_pair[1], pack.kv_heads, pack.head_dim, c_states["k_state"])
        cv2 = cand.hif4_dynamic_quantize_v(v_pair[0], v_pair[1], pack.kv_heads, pack.head_dim, c_states["v_state"])
        cq1, ck1, cv1 = first_pass[case.case_id]
        _same(cq2, cq1, f"reverse L{case.layer} w{case.test_window} q")
        _same(ck2, ck1, f"reverse L{case.layer} w{case.test_window} k")
        _same(cv2, cv1, f"reverse L{case.layer} w{case.test_window} v")
    print("reverse-order replay done")

    # Linear-side mechanism reachability: calibrate one shard of the
    # candidate and require non-empty weight states with finite params.
    lpack = v3.prepare_shard(raw, args.shard, "linear", ood=False)
    weight_states, _, _, _ = v3._calibrate(cand, lpack, device)
    n_nonempty = 0
    for (layer, role), (state, params) in sorted(weight_states.items()):
        if state:
            n_nonempty += 1
        if n_nonempty == 1:
            for key, value in params.items():
                if torch.is_tensor(value):
                    if not torch.isfinite(value.float()).all():
                        BAD.append(f"reachability {layer}/{role}.{key}: non-finite")
            print(f"reachability sample: layer {layer} role {role} state_keys={sorted(map(str, state.keys()))[:6]}")
    print(f"linear reachability: {n_nonempty}/{len(weight_states)} weight states non-empty")
    if n_nonempty == 0:
        BAD.append("reachability: no non-empty weight state — mechanism not reachable")

    print("CONTROL FAILURES:", len(BAD))
    for item in BAD:
        print("  -", item)
    return 1 if BAD else 0


if __name__ == "__main__":
    sys.exit(main())
