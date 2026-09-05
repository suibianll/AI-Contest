"""P3 local bitwise controls for the six generated probes.

Attention probes (A10 / A01) are compared operand-by-operand against the
archives that define their expected behaviour, using the probe's own
calibration states and identical per-case inputs:

- A10 (built on v164): Q/K must equal the v164 (v160 path) Q/K outputs;
  V must equal the v162 (standard codec) V output.
- A01 (built on v164): Q/K must equal the v162 standard Q/K outputs;
  V must equal the v164 (v160 path) V output.

Linear-bucket probes (W0..W3, built on v163) are checked by routing:
target-bucket weight states keep the v160 (non-empty) activation state and a
non-standard encoding; off-target buckets return empty states and encode
standard HiF4 identical to v162.

Usage: python workbench/p3_local_control.py --probe a10|a01|w0|w1|w2|w3
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "evaluator"))

import official_eval as v2  # noqa: E402
import proxy_v3_eval as v3  # noqa: E402

SOL = ROOT / "solutions"
CACHE = ROOT / "artifacts" / "official_eval" / "cache" / "qwen2.5-0.5b-proxy-v2.pt"
V162 = SOL / "20260903_v162_standard-baseline-both_scoreNA_timeNA" / "solution.py"
V163 = SOL / "20260903_v163_v160-linear_standard-attn_scoreNA_timeNA" / "solution.py"
V164 = SOL / "20260903_v164_standard-linear_v160-attn_scoreNA_timeNA" / "solution.py"
PROBES = {
    "a10": SOL / "20260905_p3a_a10_qk-v160_v-std_probe" / "solution.py",
    "a01": SOL / "20260905_p3a_a01_qk-std_v-v160_probe" / "solution.py",
    "w0": SOL / "20260905_p3c_w0_linear-bucket_probe" / "solution.py",
    "w1": SOL / "20260905_p3c_w1_linear-bucket_probe" / "solution.py",
    "w2": SOL / "20260905_p3c_w2_linear-bucket_probe" / "solution.py",
    "w3": SOL / "20260905_p3c_w3_linear-bucket_probe" / "solution.py",
}

BAD = []


def _load(path: Path) -> Any:
    return v2.load_solution(path.resolve())


def _same(a: Any, b: Any, label: str) -> None:
    if isinstance(a, dict):
        assert set(a) == set(b), f"{label}: key mismatch"
        for key in a:
            _same(a[key], b[key], f"{label}.{key}")
        return
    if torch.is_tensor(a):
        if not torch.equal(a.detach().cpu(), b.detach().cpu()):
            diff = float((a.float() - b.float()).abs().max())
            BAD.append(f"{label}: max abs diff {diff}")
        return
    if a != b:
        BAD.append(f"{label}: {a} != {b}")


def _bucket(rows: int, cols: int) -> str:
    if rows <= 256:
        return "W0"
    ratio = rows / cols
    if 0.75 <= ratio <= 1.33:
        return "W1"
    if ratio > 1.33:
        return "W2"
    return "W3"


def check_attention(probe_name: str, device: torch.device) -> None:
    probe_mod = _load(PROBES[probe_name])
    raw = v2.load_pack(CACHE)
    pack = v3.prepare_shard(raw, 0, "attention", ood=False)
    _, attention_states, _, _ = v3._calibrate(probe_mod, pack, device)
    ref_mod = _load(V164 if probe_name == "a10" else V162)
    std_mod = _load(V162 if probe_name == "a10" else V164)
    for case in pack.attention_cases:
        states = attention_states[case.layer]
        pairs = pack.test_qkv[case.test_window][case.layer]
        q_pair, k_pair, value_pair = (v2._move_pair(pair, device) for pair in pairs)
        heads = (pack.q_heads, pack.kv_heads, pack.head_dim)
        qh, kh, vh = (
            probe_mod.hif4_dynamic_quantize_q(q_pair[0], q_pair[1], pack.q_heads, pack.head_dim, states["q_state"]),
            probe_mod.hif4_dynamic_quantize_k(k_pair[0], k_pair[1], pack.kv_heads, pack.head_dim, states["k_state"]),
            probe_mod.hif4_dynamic_quantize_v(value_pair[0], value_pair[1], pack.kv_heads, pack.head_dim, states["v_state"]),
        )
        _same(qh, ref_mod.hif4_dynamic_quantize_q(q_pair[0], q_pair[1], pack.q_heads, pack.head_dim, states["q_state"]), f"{probe_name} L{case.layer} q")
        _same(kh, ref_mod.hif4_dynamic_quantize_k(k_pair[0], k_pair[1], pack.kv_heads, pack.head_dim, states["k_state"]), f"{probe_name} L{case.layer} k")
        _same(vh, std_mod.hif4_dynamic_quantize_v(value_pair[0], value_pair[1], pack.kv_heads, pack.head_dim, states["v_state"]), f"{probe_name} L{case.layer} v")
        print(f"{probe_name} L{case.layer} case {case.case_id}: q/k vs {ref_mod.__name__ if hasattr(ref_mod,'__name__') else 'ref'}, v vs std -> ok")


def check_linear(probe_name: str, device: torch.device) -> None:
    probe_mod = _load(PROBES[probe_name])
    target = probe_name.upper()
    raw = v2.load_pack(CACHE)
    pack = v3.prepare_shard(raw, 0, "linear", ood=False)
    weight_states, _, _, _ = v3._calibrate(probe_mod, pack, device)
    v162_mod = _load(V162)
    n_target = n_std = 0
    for (layer, role), (state, params) in sorted(weight_states.items()):
        w_pair = pack.weights[layer][role]
        shape = v2.dequantize_nvfp4(*w_pair).shape
        expected = _bucket(int(shape[0]), int(shape[1]))
        if expected == target:
            n_target += 1
            if not state:
                BAD.append(f"{probe_name} {layer}/{role}: expected non-empty v160 state in {expected}")
        else:
            n_std += 1
            if state:
                BAD.append(f"{probe_name} {layer}/{role}: expected empty state off-bucket")
            std_params = v162_mod.hif4_calibration_and_quantize_weight(w_pair[0], w_pair[1], [])
            _same(params, std_params["weight_params"], f"{probe_name} {layer}/{role} std params")
    print(f"{probe_name}: target-bucket states={n_target}, std states={n_std}")
    if n_target == 0:
        BAD.append(f"{probe_name}: no target-bucket state observed on local panel")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe", choices=list(PROBES), required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    device = torch.device(args.device)
    if args.probe in ("a10", "a01"):
        check_attention(args.probe, device)
    else:
        check_linear(args.probe, device)
    print("CONTROL FAILURES:", len(BAD))
    for item in BAD:
        print("  -", item)
    return 1 if BAD else 0


if __name__ == "__main__":
    sys.exit(main())
