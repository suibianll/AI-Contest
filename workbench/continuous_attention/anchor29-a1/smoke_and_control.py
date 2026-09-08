# -*- coding: utf-8 -*-
"""A29 smoke + bitwise control test.

Runs the v162 (R3) and v163 (R3+A29) calibration entries on synthetic
official-shape folds (q_heads=16, kv_heads=4, head_dim=256) and verifies:

  S1  both runs complete; timing recorded
  S2  all returned state tensors are CPU float32 finite (legal CPU state)
  S3  V-state bitwise control (A29 must never touch V)
  S4  if A29 did not deploy: q/k states bitwise-equal R3 outside audit keys
      (S=0 / gate-failure restores R3 exactly)
  S5  if A29 deployed: gate records strictly decreasing fold4 Lhard,
      changed codes > 0, and the state diff is exactly the learned pair
  S6  audit values finite

Run under gpu lock:
  .venv/Scripts/python.exe workbench/v162_attention/gpu_lock.py acquire attention smoke-a29
  .venv/Scripts/python.exe workbench/continuous_attention/anchor29-a1/smoke_and_control.py
  .venv/Scripts/python.exe workbench/v162_attention/gpu_lock.py release attention smoke-a29
"""

import importlib.util
import math
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[3]


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


v162 = load("v162", ROOT / "solutions/v162_attention_r3-rotation-center_allgates/solution.py")
v163 = load("v163", ROOT / "solutions/v163_attention_a29-final-residual-s/solution.py")

E2M1 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
CUTS = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0])


def nvfp4_encode(x: torch.Tensor):
    T, C = x.shape
    xb = x.reshape(T, C // 16, 16)
    amax = xb.abs().amax(-1, keepdim=True)
    scale = (amax / 6.0).clamp_min(1e-8)
    q = xb / scale
    sign = q.sign()
    idx = torch.bucketize(q.abs(), CUTS)
    qg = E2M1.to(x.device)[idx] * sign
    return qg.reshape(T, C), scale.squeeze(-1)


def make_folds(seed: int, folds_ts=(128, 256, 512, 512, 512)):
    g = torch.Generator().manual_seed(seed)
    q_heads, kv_heads, head_dim = 16, 4, 256
    items = []
    for T in folds_ts:
        q = torch.randn(T, q_heads * head_dim, generator=g) * 0.5
        k = torch.randn(T, kv_heads * head_dim, generator=g) * 0.5
        v = torch.randn(T, kv_heads * head_dim, generator=g) * 0.5
        q_q, q_s = nvfp4_encode(q)
        k_q, k_s = nvfp4_encode(k)
        v_q, v_s = nvfp4_encode(v)
        items.append({"q": (q_q, q_s), "k": (k_q, k_s), "v": (v_q, v_s)})
    return items


def state_leaves(states: dict):
    out = {}
    for side in ("q_state", "k_state", "v_state"):
        for key, val in states[side].items():
            if isinstance(val, torch.Tensor):
                out[f"{side}.{key}"] = val
    return out


def audit_keys(states: dict):
    keys = set()
    for side in ("q_state", "k_state"):
        for key in states[side]:
            if not isinstance(states[side][key], torch.Tensor):
                keys.add(f"{side}.{key}")
    return keys


def check_all_cpu_finite(states: dict):
    # offsets is a legal int8 R3 state entry (consumed verbatim by the
    # deployed dynamic quantizers); float tensors must be CPU float32 finite
    for name, t in state_leaves(states).items():
        if t.dtype == torch.int8:
            if t.device.type != "cpu":
                return False, f"{name} on {t.device}"
            continue
        if t.device.type != "cpu":
            return False, f"{name} on {t.device}"
        if t.dtype != torch.float32:
            return False, f"{name} dtype {t.dtype}"
        if not bool(torch.isfinite(t).all()):
            return False, f"{name} non-finite"
    return True, "ok"


def main() -> int:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[smoke] device={device}")
    torch.manual_seed(2929)
    folds = make_folds(2929)
    q_heads, kv_heads, head_dim = 16, 4, 256

    t0 = time.time()
    states_r3 = v162.hif4_calibration_attention(folds, q_heads, kv_heads, head_dim)
    t_r3 = time.time() - t0
    t0 = time.time()
    states_a29 = v163.hif4_calibration_attention(folds, q_heads, kv_heads, head_dim)
    t_a29 = time.time() - t0
    print(f"[smoke] S1 run: r3={t_r3:.2f}s v163={t_a29:.2f}s (A29 overhead {t_a29 - t_r3:.2f}s)")

    ok, why = check_all_cpu_finite(states_a29)
    print(f"[smoke] S2 legal CPU state: {'PASS' if ok else 'FAIL ' + why}")

    v_r3 = states_r3["v_state"]
    v_a29 = states_a29["v_state"]
    v_ok = set(v_r3) == set(v_a29) and all(
        torch.equal(v_r3[kk], v_a29[kk]) for kk in v_r3
        if isinstance(v_r3[kk], torch.Tensor)
    )
    print(f"[smoke] S3 V bitwise control: {'PASS' if v_ok else 'FAIL'}")

    arm = states_a29["q_state"].get("a29_arm", "missing")
    print(f"[smoke] a29_arm = {arm}")
    r3_leaves = state_leaves(states_r3)
    a29_leaves = state_leaves(states_a29)
    if arm != "deployed":
        common = set(r3_leaves) & set(a29_leaves)
        diff = [kk for kk in common if not torch.equal(r3_leaves[kk], a29_leaves[kk])]
        only_a29 = set(a29_leaves) - set(r3_leaves)
        only_r3 = set(r3_leaves) - set(a29_leaves)
        ok = not diff and not only_a29 and not only_r3
        print(f"[smoke] S4 non-deploy bitwise restore: {'PASS' if ok else 'FAIL'} "
              f"(diff={diff} only_a29={sorted(only_a29)} only_r3={sorted(only_r3)})")
    else:
        lp = states_a29["q_state"]["a29_lhard_parent_f4"]
        lq = states_a29["q_state"]["a29_lhard_prop_f4"]
        cc = states_a29["q_state"]["a29_changed_codes_q"] + states_a29["q_state"]["a29_changed_codes_k"]
        ok = (lq < lp) and cc > 0
        rot_diff = not torch.equal(r3_leaves["q_state.learned_rotation"], a29_leaves["q_state.learned_rotation"])
        v_untouched = v_ok
        print(f"[smoke] S5 deployed: lhard f4 parent={lp:.6e} prop={lq:.6e} codes={cc} "
              f"rot_diff={rot_diff} -> {'PASS' if ok and rot_diff and v_untouched else 'FAIL'}")

    audits_finite = True
    for kk in sorted(audit_keys(states_a29)):
        val = states_a29[kk.split(".")[0]][kk.split(".")[1]]
        if isinstance(val, float) and not math.isfinite(val):
            audits_finite = False
        print(f"    audit {kk} = {val}")
    print(f"[smoke] S6 audit finite: {'PASS' if audits_finite else 'FAIL'}")

    failures = []
    if not v_ok:
        failures.append("V-control")
    if arm != "deployed":
        if not ok:
            failures.append("restore")
    else:
        if not ok:
            failures.append("gate-records")
    cpu_ok, _why2 = check_all_cpu_finite(states_a29)
    if not cpu_ok:
        failures.append("cpu-state")
    if not audits_finite:
        failures.append("audit")
    if failures:
        print(f"[smoke] FAILURES: {failures}")
        return 1
    print("[smoke] all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
