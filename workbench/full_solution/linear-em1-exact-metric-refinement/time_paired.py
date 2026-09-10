"""Paired in-process timing of the L-EM1 dynamic API against the parent root.

The shard0 record times the parent and the candidate in two different
processes, so its per-call delta mixes mechanism cost with machine load.  This
script loads both calibration artifacts (cache hits, no calibration run), then
alternates parent/candidate calls on the same case with ``cuda.synchronize``
inside the timed region, exactly like ``official_eval.py`` times the six APIs.

Read-only with respect to the repo: it writes only ``timing_paired.json`` next
to itself.
"""

from __future__ import annotations

import gc
import json
import statistics
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
REPEATS = 8
WARMUP = 2
CASES = (
    (0, "q"),
    (0, "o"),
    (0, "fc_gate"),
    (0, "proj"),
    (6, "q"),
    (12, "v"),
)


def main() -> int:
    device = torch.device("cuda")
    raw = v2.load_pack(DENSE_CACHE)
    pack = p3.prepare_shard(raw, 0, "linear", False)
    del raw
    gc.collect()

    solutions: dict[str, object] = {}
    states: dict[str, dict] = {}
    for label, path in (("parent", PARENT), ("candidate", CANDIDATE)):
        identity = p3._calibration_identity(path, pack, device)
        cache = p3.default_calibration_cache_path(identity)
        if not cache.is_file():
            raise SystemExit(f"missing calibration cache for {label}: {cache}")
        solution = v2.load_solution(path)
        weight_states, _ = p3.load_calibration_artifact(cache, identity, pack)
        solutions[label] = solution
        states[label] = weight_states
        print(
            f"[{label}] {v2.sha256_file(path)[:16]} states={len(weight_states)} "
            f"cache={cache.name}",
            flush=True,
        )

    by_key = {(case.layer, case.role): case for case in pack.linear_cases}
    report = {"repeats": REPEATS, "warmup": WARMUP, "device": str(device), "cases": []}
    for layer, role in CASES:
        case = by_key[(layer, role)]
        pair = v2._move_pair(
            pack.test_activations[role][case.test_window][layer], device
        )
        state = {label: states[label][(layer, role)][0] for label in states}
        timings: dict[str, list[float]] = {"parent": [], "candidate": []}
        outputs: dict[str, tuple] = {}
        for label in ("parent", "candidate"):
            for _ in range(WARMUP):
                with torch.no_grad():
                    outputs[label] = solutions[label].hif4_dynamic_quantize_activation(
                        pair[0], pair[1], state[label]
                    )
                torch.cuda.synchronize(device)
        for _ in range(REPEATS):
            for label in ("parent", "candidate"):
                torch.cuda.synchronize(device)
                started = time.perf_counter()
                with torch.no_grad():
                    solutions[label].hif4_dynamic_quantize_activation(
                        pair[0], pair[1], state[label]
                    )
                torch.cuda.synchronize(device)
                timings[label].append(time.perf_counter() - started)

        parent_out = v2._cpu_params(outputs["parent"])
        candidate_out = v2._cpu_params(outputs["candidate"])
        changed = int((parent_out["mant"] != candidate_out["mant"]).sum())
        parent_med = statistics.median(timings["parent"])
        candidate_med = statistics.median(timings["candidate"])
        entry = {
            "layer": layer,
            "role": role,
            "channels": int(pack.weights[layer][role][0].shape[0]),
            "rows": int(pair[0].shape[0]),
            "changed_mantissa": changed,
            "parent_median_seconds": parent_med,
            "candidate_median_seconds": candidate_med,
            "delta_median_seconds": candidate_med - parent_med,
            "parent_timings": timings["parent"],
            "candidate_timings": timings["candidate"],
        }
        report["cases"].append(entry)
        print(
            f"layer{layer}/{role:<8} rows={entry['rows']:<4} n={entry['channels']:<5} "
            f"parent={parent_med:.4f}s candidate={candidate_med:.4f}s "
            f"delta={candidate_med - parent_med:+.4f}s changed_mant={changed}",
            flush=True,
        )

    deltas = [entry["delta_median_seconds"] for entry in report["cases"]]
    report["delta_median_mean_seconds"] = statistics.mean(deltas)
    report["delta_median_worst_seconds"] = max(deltas)
    report["official_linear_calls"] = 168
    report["projected_official_added_seconds"] = (
        statistics.mean(deltas) * report["official_linear_calls"]
    )
    (HERE / "timing_paired.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(
        f"\nmean delta {report['delta_median_mean_seconds']:+.4f}s/call; "
        f"projected added on 168 official Linear calls: "
        f"{report['projected_official_added_seconds']:+.1f}s",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
