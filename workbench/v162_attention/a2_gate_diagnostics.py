"""A2 gate diagnostics: train the per-KV-group rotation on real calibration data.

Runs the deployed candidate calibration path (arm="gate") on the small panel
layers 0/8/15/23 with the real proxy-v2 cache windows, and records the gate
losses of the identity / fixed-H / learned arms plus the deployed decision.
This is diagnostics only; the four-arm ranking evidence comes from the
evaluator runs on arm variants.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "evaluator"))
import official_eval as v2  # noqa: E402


def load_candidate():
    spec = importlib.util.spec_from_file_location(
        "v162_attention_candidate_gate",
        ROOT / "workbench/v162_attention/candidate/solution.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> int:
    layers = [0, 8, 15, 23]
    cache_path = ROOT / "artifacts/official_eval/cache/qwen2.5-0.5b-proxy-v2.pt"
    print("loading dense cache ...", flush=True)
    pack = torch.load(cache_path, map_location="cpu", weights_only=False)
    q_heads, kv_heads, head_dim = pack["q_heads"], pack["kv_heads"], pack["head_dim"]
    candidate = load_candidate()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    report = {
        "device": str(device),
        "q_heads": q_heads, "kv_heads": kv_heads, "head_dim": head_dim,
        "calibration_window_lengths": [len(w["input_ids"]) for w in pack["calibration_windows"]],
        "layers": {},
    }
    for layer in layers:
        calib = []
        for sample in range(len(pack["calibration_windows"])):
            q, k, v = pack["calibration_qkv"][sample][layer]
            item = {"q": v2._pair(q), "k": v2._pair(k), "v": v2._pair(v)}
            calib.append(v2._move_qkv(item, device))
        started = time.perf_counter()
        states = candidate.hif4_calibration_attention(calib, q_heads, kv_heads, head_dim)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - started
        q_state = states["q_state"]
        entry = {
            "calibration_seconds": round(elapsed, 3),
            "deployed_arm": q_state.get("arm"),
            "mode": q_state.get("mode"),
            "gate_loss_identity": q_state.get("gate_loss_identity"),
            "gate_loss_h": q_state.get("gate_loss_h"),
            "gate_loss_learned": q_state.get("gate_loss_learned"),
            "trained_steps": q_state.get("trained_steps"),
        }
        report["layers"][str(layer)] = entry
        print(
            f"layer {layer}: deployed={entry['deployed_arm']} "
            f"I={entry['gate_loss_identity']:.6f} H={entry['gate_loss_h']:.6f} "
            f"learned={entry['gate_loss_learned']:.6f} calib={elapsed:.1f}s",
            flush=True,
        )
    out = Path(__file__).resolve().parent / "research" / "a2_gate_diagnostics.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"saved {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
