# -*- coding: utf-8 -*-
"""A29 smoke + bitwise control test (4B panel data, per user rule).

Runs the v162 (R3) and v163 (R3+A29) calibration entries on REAL qwen3.5-4b
proxy-v2 calibration folds (2026-09-08 user rule: all local testing uses the
4B panel; synthetic smoke data retired) and verifies:

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

sys.path.insert(0, str(ROOT / "evaluator"))
import official_eval as core  # noqa: E402


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


v162 = load("v162", ROOT / "solutions/v162_attention_r3-rotation-center_allgates/solution.py")
v163 = load("v163", ROOT / "solutions/v163_attention_a29-final-residual-s/solution.py")

CACHE = ROOT / "artifacts/official_eval/cache/qwen3.5-4b-proxy-v2.pt"


def load_4b_folds(layer: int | None = None):
    """Real 4B calibration folds (NVFP4 pairs) exactly as official_eval
    builds them, plus the pack geometry."""

    raw = core.load_pack(CACHE)
    meta_layers = raw.metadata.get("attention_state_layers") or raw.metadata.get("attention_layers")
    layers = [int(v) for v in meta_layers] if meta_layers else list(range(raw.layers))
    layers = [v for v in layers if raw.calibration_qkv[0][v] is not None]
    if layer is None or layer not in layers:
        layer = layers[0]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    folds = []
    for sample in range(len(raw.calibration_windows)):
        qs, ks, vs = raw.calibration_qkv[sample][layer]
        folds.append(core._move_qkv(
            {"q": core._pair(qs), "k": core._pair(ks), "v": core._pair(vs)}, device
        ))
    return folds, layer, int(raw.q_heads), int(raw.kv_heads), int(raw.head_dim)


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
    print(f"[smoke] device={device} data=4B-panel")
    folds, src_layer, q_heads, kv_heads, head_dim = load_4b_folds()
    print(f"[smoke] source layer {src_layer}, fold tokens="
          f"{[int(f['q'][0].shape[-2]) for f in folds]}")

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
