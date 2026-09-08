"""A29 gate_check on the real qwen3.5-4b proxy-v2 calibration data.

Per plan A-R1: run the v163 (R3 + A29) calibration_attention entry per
attention state layer on the real NVFP4 calibration folds, against the v162
R3 parent, and record the A29 audit statistics (CG residuals, S norm, alpha,
changed codes, fold3/fold4 hard-output MSE, deploy decision) plus API time.

Usage (GPU lock held externally):
  .venv/Scripts/python.exe workbench/v162_attention/gpu_lock.py acquire attention a29-gate4b
  .venv/Scripts/python.exe workbench/continuous_attention/anchor29-a1/gate_check_4b.py
  .venv/Scripts/python.exe workbench/v162_attention/gpu_lock.py release attention a29-gate4b
"""

from __future__ import annotations

import importlib.util
import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[3]
CACHE = ROOT / "artifacts/official_eval/cache/qwen3.5-4b-proxy-v2.pt"
OUT = Path(__file__).resolve().parent / "gate_check_4b.json"

sys.path.insert(0, str(ROOT / "evaluator"))
import official_eval as core  # noqa: E402


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


v162 = load("v162", ROOT / "solutions/v162_attention_r3-rotation-center_allgates/solution.py")
v163 = load("v163", ROOT / "solutions/v163_attention_a29-final-residual-s/solution.py")


def state_diff_count(a: dict, b: dict) -> int:
    total = 0
    for side in ("q_state", "k_state", "v_state"):
        for key, val in a[side].items():
            if isinstance(val, torch.Tensor):
                other = b[side].get(key)
                if other is None or not torch.equal(val, other):
                    total += 1
    return total


def main() -> int:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[gate4b] device={device} cache={CACHE.name}")
    raw = core.load_pack(CACHE)
    meta_layers = raw.metadata.get("attention_state_layers") or raw.metadata.get("attention_layers")
    layers = [int(x) for x in meta_layers] if meta_layers else list(range(raw.layers))
    layers = [lay for lay in layers if raw.calibration_qkv[0][lay] is not None]
    print(f"[gate4b] layers={layers} q_heads={raw.q_heads} kv_heads={raw.kv_heads} head_dim={raw.head_dim}")
    print(f"[gate4b] calibration fold lengths={[len(w.input_ids) for w in raw.calibration_windows]}")

    rows = []
    for layer in layers:
        q, k, v = raw.calibration_qkv[0][layer]
        # per-sample calibration exactly as official_eval prepare_pack builds it
        calibration = []
        for sample in range(len(raw.calibration_windows)):
            qs, ks, vs = raw.calibration_qkv[sample][layer]
            calibration.append(core._move_qkv(
                {"q": core._pair(qs), "k": core._pair(ks), "v": core._pair(vs)}, device
            ))
        lengths = [int(item["q"][0].shape[-2]) if item["q"][0].dim() >= 2 else -1 for item in calibration]

        t0 = time.time()
        states_r3 = v162.hif4_calibration_attention(calibration, raw.q_heads, raw.kv_heads, raw.head_dim)
        t_r3 = time.time() - t0
        t0 = time.time()
        states_a29 = v163.hif4_calibration_attention(calibration, raw.q_heads, raw.kv_heads, raw.head_dim)
        t_a29 = time.time() - t0

        aud = states_a29["q_state"]
        cg = aud.get("a29_cg")
        cg_summary = None
        if isinstance(cg, str):
            parsed = json.loads(cg.replace("(", "[").replace(")", "]"))
            converged = sum(1 for _, res, _ in parsed if res < 1e-3)
            breakdown = sum(1 for _, res, it in parsed if it <= 2)
            cg_summary = {"instances": len(parsed), "converged": converged, "frozen_early": breakdown}
        row = {
            "layer": layer,
            "layer_type": (raw.metadata.get("layer_types") or ["?"] * raw.layers)[layer],
            "fold_token_counts": lengths,
            "t_r3_s": round(t_r3, 2),
            "t_a29_s": round(t_a29, 2),
            "overhead_s": round(t_a29 - t_r3, 2),
            "a29_arm": aud.get("a29_arm"),
            "a29_error": aud.get("a29_error"),
            "a29_s_fnorm": aud.get("a29_s_fnorm"),
            "a29_alpha": aud.get("a29_alpha"),
            "a29_alpha_blocks": aud.get("a29_alpha_blocks"),
            "a29_cg": cg_summary,
            "changed_codes_q": aud.get("a29_changed_codes_q"),
            "changed_codes_k": aud.get("a29_changed_codes_k"),
            "lhard_parent_f3": aud.get("a29_lhard_parent_f3"),
            "lhard_prop_f3": aud.get("a29_lhard_prop_f3"),
            "lhard_parent_f4": aud.get("a29_lhard_parent_f4"),
            "lhard_prop_f4": aud.get("a29_lhard_prop_f4"),
            "deployed": aud.get("a29_arm") == "deployed",
            "state_tensor_diffs_vs_r3": state_diff_count(states_r3, states_a29),
        }
        rows.append(row)
        print(json.dumps(row, ensure_ascii=False))
        sys.stdout.flush()

    deployed = [r["layer"] for r in rows if r["deployed"]]
    print(f"[gate4b] deployed layers: {deployed if deployed else 'NONE (all gates held parent)'}")
    OUT.write_text(json.dumps({"cache": str(CACHE), "rows": rows}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[gate4b] wrote {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
