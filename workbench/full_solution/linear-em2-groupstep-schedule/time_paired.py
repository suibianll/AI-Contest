"""Paired in-process timing of the L-EM2 dynamic API against the parent root.

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
# The plan's time-driven fallback: measure K = 2 (pre-registered) and K = 1 in
# the same process, on the same cases, so the choice is a measured one.
_EM2_PASSES = 2
ARMS = ("parent", "candidate_k2", "candidate_k1")
ARM_SOLUTION = {"parent": "parent", "candidate_k2": "candidate", "candidate_k1": "candidate"}
K_BY_ARM = {"parent": None, "candidate_k2": _EM2_PASSES, "candidate_k1": 1}
# shard0 covers layers {0, 6, 12, 18}.  Every role but `o` is in=2560, so the
# in=4096 projection rests on the `o` role alone -- and a single case left the
# two estimators 7% apart against 0.5% at in=2560.  `o` is sampled at all four
# shard0 layers so both columns of the projection have independent cases behind
# them.
CASES = (
    (0, "q"),
    (0, "o"),
    (0, "fc_gate"),
    (0, "proj"),
    (6, "q"),
    (6, "o"),
    (12, "v"),
    (12, "o"),
    (18, "o"),
)


def main() -> int:
    device = torch.device("cuda")
    raw = v2.load_pack(DENSE_CACHE)
    pack = p3.prepare_shard(raw, 0, "linear", False)
    del raw
    gc.collect()

    # Calibrate in this process rather than loading a shard artifact: the
    # candidate's identity has no cache yet, and a same-process pairing is what
    # the dynamic measurement wants anyway.  Only the CASES roles are needed.
    solutions: dict[str, object] = {}
    states: dict[str, dict] = {}
    for label, path in (("parent", PARENT), ("candidate", CANDIDATE)):
        solution = v2.load_solution(path)
        solutions[label] = solution
        states[label] = {}
        print(f"[{label}] {v2.sha256_file(path)[:16]}", flush=True)
    for layer, role in CASES:
        weight_pair = v2._move_pair(pack.weights[layer][role], device)
        calibration = [
            v2._move_pair(pack.linear_calibration_activations[role][sample][layer], device)
            for sample in pack.metadata["linear_calibration_indices"]
        ]
        for label in ("parent", "candidate"):
            with torch.no_grad():
                result = solutions[label].hif4_calibration_and_quantize_weight(
                    weight_pair[0], weight_pair[1], calibration
                )
            torch.cuda.synchronize(device)
            states[label][(layer, role)] = result["activation_state"]
            print(
                f"[{label}] calibrated layer{layer}/{role} "
                f"in={result['activation_state'].get('in_features')} "
                f"arm={result['activation_state'].get('em1_arm', '-')}",
                flush=True,
            )
            del result
        gc.collect()
        torch.cuda.empty_cache()

    by_key = {(case.layer, case.role): case for case in pack.linear_cases}
    report = {"repeats": REPEATS, "warmup": WARMUP, "device": str(device), "cases": []}
    for layer, role in CASES:
        case = by_key[(layer, role)]
        pair = v2._move_pair(
            pack.test_activations[role][case.test_window][layer], device
        )
        state = {label: states[label][(layer, role)] for label in states}
        timings: dict[str, list[float]] = {arm: [] for arm in ARMS}
        outputs: dict[str, tuple] = {}
        for arm in ARMS:
            passes = K_BY_ARM[arm]
            if passes is not None:
                solutions["candidate"]._EM1_PASSES = passes
            for _ in range(WARMUP):
                with torch.no_grad():
                    outputs[arm] = solutions[ARM_SOLUTION[arm]].hif4_dynamic_quantize_activation(
                        pair[0], pair[1], state[ARM_SOLUTION[arm]]
                    )
                torch.cuda.synchronize(device)
        for _ in range(REPEATS):
            for arm in ARMS:
                passes = K_BY_ARM[arm]
                if passes is not None:
                    solutions["candidate"]._EM1_PASSES = passes
                torch.cuda.synchronize(device)
                started = time.perf_counter()
                with torch.no_grad():
                    solutions[ARM_SOLUTION[arm]].hif4_dynamic_quantize_activation(
                        pair[0], pair[1], state[ARM_SOLUTION[arm]]
                    )
                torch.cuda.synchronize(device)
                timings[arm].append(time.perf_counter() - started)

        parent_out = v2._cpu_params(outputs["parent"])
        candidate_out = v2._cpu_params(outputs["candidate_k2"])
        changed = int((parent_out["mant"] != candidate_out["mant"]).sum())
        solutions["candidate"]._EM1_PASSES = _EM2_PASSES
        # The box is shared with a desktop, so a per-repeat spread of 0.02-0.07 s
        # is routine against deltas of the same size.  The median is the honest
        # central estimate; the minimum is the least-contended one, and the two
        # bracketing the same answer is the evidence the delta is real rather
        # than load.  Both are reported and both are projected.
        parent_med = statistics.median(timings["parent"])
        parent_min = min(timings["parent"])
        medians = {arm: statistics.median(timings[arm]) for arm in ARMS}
        minima = {arm: min(timings[arm]) for arm in ARMS}
        entry = {
            "layer": layer,
            "role": role,
            "in_features": int(state["candidate"].get("in_features", -1)),
            "rows": int(pair[0].shape[0]),
            "changed_mantissa": changed,
            "parent_median_seconds": parent_med,
            "median_seconds": medians,
            "delta_median_seconds": medians["candidate_k2"] - parent_med,
            "delta_k1_median_seconds": medians["candidate_k1"] - parent_med,
            "delta_min_seconds": minima["candidate_k2"] - parent_min,
            "delta_k1_min_seconds": minima["candidate_k1"] - parent_min,
            "spread_seconds": {
                arm: max(timings[arm]) - min(timings[arm]) for arm in ARMS
            },
            "timings": timings,
        }
        report["cases"].append(entry)
        print(
            f"layer{layer}/{role:<8} rows={entry['rows']:<4} in={entry['in_features']:<5} "
            f"parent={parent_med:.4f}s k2={medians['candidate_k2']:.4f}s "
            f"delta_k2={medians['candidate_k2'] - parent_med:+.4f}s "
            f"delta_k1={medians['candidate_k1'] - parent_med:+.4f}s "
            f"changed_mant={changed}",
            flush=True,
        )

    # Official Linear panel: 24 layers x 7 roles, one dynamic call each; the
    # mechanism is in scope for every role but proj (in_features = 9216), so
    # 120 calls at in=2560 and 24 at in=4096.  The per-in_features means are the
    # projection basis -- a flat mean over the case list would mis-weight them.
    official_by_channels = {2560: 120, 4096: 24}
    for key in (
        "delta_median_seconds",
        "delta_k1_median_seconds",
        "delta_min_seconds",
        "delta_k1_min_seconds",
    ):
        scoped = [e for e in report["cases"] if 0 < e["in_features"] <= 4096]
        per_channels = {
            channels: statistics.mean(
                [e[key] for e in scoped if e["in_features"] == channels]
            )
            for channels in sorted({e["in_features"] for e in scoped})
        }
        projected = sum(
            per_channels[channels] * count
            for channels, count in official_by_channels.items()
        )
        report[f"mean_{key}_by_in_features"] = per_channels
        report[f"projected_{key}"] = projected
        print(
            f"\n[{key}] "
            + ", ".join(
                f"in={ch} {value:+.4f}s/call" for ch, value in per_channels.items()
            )
            + f"\n  projected on 120x2560 + 24x4096 official Linear calls: {projected:+.1f}s",
            flush=True,
        )
    report["official_linear_calls_by_in_features"] = official_by_channels
    report["official_linear_calls"] = 144
    report["official_linear_calls_out_of_scope"] = 24
    worst = max(e["delta_k1_median_seconds"] for e in report["cases"])
    report["delta_k1_median_worst_seconds"] = worst
    # Out-of-scope control: proj (in=9216) takes the bypass, so any non-zero
    # delta there is pure measurement noise.  Its magnitude is the noise floor
    # the in-scope deltas must clear.
    report["noise_floor_seconds"] = {
        key: next(
            e[key] for e in report["cases"] if e["role"] == "proj"
        )
        for key in ("delta_median_seconds", "delta_k1_median_seconds")
    }
    (HERE / "timing_paired.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
