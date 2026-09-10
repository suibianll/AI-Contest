"""Paired in-process timing of the calibration API for L-EM1 against the root.

``time_paired.py`` measures the *dynamic* API only.  The official judge times
the six candidate APIs, so the L-EM1 compile hook (``H`` build, Cholesky
inverse, fp32 state copy) is charged once per (layer, role) as well -- 168
calibration calls on Qwen, 144 of them in scope.  This script runs the parent
calibration and the L-EM1 calibration on the same (layer, role) pairs in one
process and reports the per-call delta, exactly like the dynamic pairing.

Read-only with respect to the repo: it writes only ``timing_calibration.json``
next to itself.
"""

from __future__ import annotations

import gc
import json
import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "evaluator"))

import official_eval as v2  # noqa: E402
import proxy_v3_eval as p3  # noqa: E402

PARENT = ROOT / "solution.py"
CANDIDATE = HERE / "candidate" / "solution.py"
DENSE_CACHE = v2.CACHE_DIR / "qwen3.5-4b-proxy-v2.pt"
CASES = ((0, "q"), (0, "o"), (0, "proj"), (0, "k"), (0, "fc_gate"))


def main() -> int:
    device = torch.device("cuda")
    raw = v2.load_pack(DENSE_CACHE)
    pack = p3.prepare_shard(raw, 0, "linear", False)
    del raw
    gc.collect()

    solutions = {
        "parent": v2.load_solution(PARENT),
        "candidate": v2.load_solution(CANDIDATE),
    }
    report = {"device": str(device), "cases": []}
    for layer, role in CASES:
        weight_pair = v2._move_pair(pack.weights[layer][role], device)
        calibration = [
            v2._move_pair(pack.linear_calibration_activations[role][sample][layer], device)
            for sample in pack.metadata["linear_calibration_indices"]
        ]
        row = {"layer": layer, "role": role}
        for label in ("parent", "candidate"):
            with torch.no_grad():
                solutions[label].hif4_calibration_and_quantize_weight(
                    weight_pair[0], weight_pair[1], calibration
                )
            torch.cuda.synchronize(device)
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.synchronize(device)
            started = time.perf_counter()
            with torch.no_grad():
                result = solutions[label].hif4_calibration_and_quantize_weight(
                    weight_pair[0], weight_pair[1], calibration
                )
            torch.cuda.synchronize(device)
            row[f"{label}_seconds"] = time.perf_counter() - started
            state = result["activation_state"]
            row[f"{label}_channels"] = int(state.get("in_features", -1))
            del result, state
            gc.collect()
            torch.cuda.empty_cache()
        row["in_features"] = row["parent_channels"]
        row["delta_seconds"] = row["candidate_seconds"] - row["parent_seconds"]
        report["cases"].append(row)
        print(
            f"layer{layer}/{row['role']:<8} in={row['parent_channels']:<5} "
            f"parent={row['parent_seconds']:.4f}s candidate={row['candidate_seconds']:.4f}s "
            f"delta={row['delta_seconds']:+.4f}s",
            flush=True,
        )

    scoped = [row for row in report["cases"] if row["parent_channels"] <= 4096]
    per_channel_2560 = [r["delta_seconds"] for r in scoped if r["parent_channels"] == 2560]
    per_channel_4096 = [r["delta_seconds"] for r in scoped if r["parent_channels"] == 4096]
    mean_2560 = sum(per_channel_2560) / len(per_channel_2560)
    mean_4096 = sum(per_channel_4096) / len(per_channel_4096)
    official_scoped = 24 * 5 * mean_2560 + 24 * 1 * mean_4096
    report["mean_delta_in2560_seconds"] = mean_2560
    report["mean_delta_in4096_seconds"] = mean_4096
    report["official_scoped_calibrations"] = 24 * 6
    report["projected_official_added_seconds"] = official_scoped
    (HERE / "timing_calibration.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(
        f"\nin=2560 delta {mean_2560:+.4f}s/call, in=4096 delta {mean_4096:+.4f}s/call; "
        f"projected added on 144 official in-scope calibrations: {official_scoped:+.1f}s",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
