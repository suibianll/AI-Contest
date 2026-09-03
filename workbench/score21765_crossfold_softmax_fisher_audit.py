"""Run the 21765-A Fisher estimability audit on the cached compact states."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
import time
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
EVALUATOR = ROOT / "evaluator"
sys.path.insert(0, str(EVALUATOR))

import official_eval  # noqa: E402
from reference_hif4 import validate_state  # noqa: E402


CANDIDATE = ROOT / "workbench/score21765_crossfold_softmax_fisher.py"
DEFAULT_OUTPUT = ROOT / "artifacts/official_eval/score21765-a0-softmax-fisher-audit.json"


def load_candidate():
    spec = importlib.util.spec_from_file_location("score21765_a0", CANDIDATE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the registered A0 audit")
    profile = official_eval._nvfp4_cache_profile(
        linear_count=None,
        attention_count=None,
        full_cases=False,
        effect_panel=False,
        compact_panel=True,
        evaluation_scenario="attention",
    )
    source_cache = official_eval.DEFAULT_CACHE
    nvfp4_cache = official_eval._default_nvfp4_cache_path(source_cache, profile)
    pack = official_eval.load_nvfp4_cache(nvfp4_cache, source_cache, profile)
    candidate = load_candidate()
    candidate._ATTN_CROSSFOLD_FISHER_AUDIT.clear()
    layers = [int(layer) for layer in pack.metadata["attention_state_layers"]]
    records = []
    total_seconds = 0.0

    for layer in layers:
        calibration = [
            official_eval._move_qkv(pack.calibration_qkv[index][layer], device)
            for index in range(len(pack.calibration_windows))
        ]
        started = time.perf_counter()
        states = candidate.hif4_calibration_attention(
            calibration, pack.q_heads, pack.kv_heads, pack.head_dim
        )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - started
        total_seconds += elapsed
        for state in states.values():
            validate_state(state)
        record = dict(candidate._ATTN_CROSSFOLD_FISHER_AUDIT[-1])
        record["layer"] = int(layer)
        record["calibration_seconds"] = float(elapsed)
        records.append(record)
        print(
            f"layer={layer} seconds={elapsed:.3f} "
            f"q_changed={record['q']['changed_channels']} "
            f"k_changed={record['k']['changed_channels']} "
            f"q_rho={record['q']['rho_median']:.6f} "
            f"k_rho={record['k']['rho_median']:.6f}",
            flush=True,
        )

    summary = {}
    for name in ("q", "k"):
        changed = [int(record[name]["changed_channels"]) for record in records]
        rho = torch.tensor(
            [float(record[name]["rho_median"]) for record in records]
        )
        summary[name] = {
            "layers_changed": int(sum(value > 0 for value in changed)),
            "changed_channels": int(sum(changed)),
            "rho_median_across_layers": float(rho.median()),
            "rho_max_across_layers": float(rho.max()),
        }
    output = {
        "experiment": "score21765-a0-crossfold-softmax-fisher",
        "scope": "attention-compact-selected-calibration-states-only",
        "candidate": str(CANDIDATE.relative_to(ROOT)),
        "candidate_sha256": hashlib.sha256(CANDIDATE.read_bytes()).hexdigest().upper(),
        "parent_sha256": "33B1D061CE6BFCD92659C597BE4830BB9B910E646FF518433DA67B925AE8680D",
        "device": str(device),
        "layers": layers,
        "total_calibration_seconds": float(total_seconds),
        "summary": summary,
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    print(f"wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
