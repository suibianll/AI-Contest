"""Aggregate the A2 four-arm + strong-control shard JSONs into the ledger table.

Reads candidate/baseline attention shard JSONs for shards 0,2,3,5 under
artifacts/proxy_v3/v162-independent/attention/<run>/ and emits paired stats:
per-arm mean/median/quartiles, per-layer and per-split means, deltas vs the
v162 zero, deltas vs the strong controls, L1_negative, and D_strong.
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "artifacts/proxy_v3/v162-independent/attention"
SHARDS = (0, 2, 3, 5)
RUNS = ("a2-gate", "a2-h", "a2-learned", "a2-strong-v168", "a2-strong-v189")


def load_cases(run: str, kind: str) -> dict[tuple, float]:
    cases: dict[tuple, float] = {}
    for shard in SHARDS:
        path = BASE / run / run / f"{kind}-attention-shard{shard}.json"
        if not path.exists():
            raise FileNotFoundError(path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        for result in payload["results"]:
            for case in result["case_scores"]["attention"]:
                key = (int(case["layer"]), int(case["test_window"]), str(case["test_split"]), int(case["test_length"]))
                if key in cases:
                    raise ValueError(f"duplicate case identity {key} in {run}/{kind}")
                cases[key] = float(case["gain"])
    return cases


def describe(values: list[float]) -> dict:
    ordered = sorted(values)
    n = len(ordered)
    q25 = ordered[int(0.25 * (n - 1))]
    q75 = ordered[int(0.75 * (n - 1))]
    worst_quartile_mean = statistics.fmean(ordered[: max(1, n // 4)])
    return {
        "n": n,
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "q25": q25,
        "q75": q75,
        "worst_quartile_mean": worst_quartile_mean,
        "positive": sum(1 for v in values if v > 0),
        "zero": sum(1 for v in values if v == 0),
        "negative": sum(1 for v in values if v < 0),
    }


def main() -> int:
    baseline = load_cases("a2-gate", "baseline")
    arms = {run: load_cases(run, "candidate") for run in RUNS}
    summary = {"case_count": len(baseline), "baseline_gain_mean": statistics.fmean(baseline.values()), "arms": {}}
    for run, cases in arms.items():
        if set(cases) != set(baseline):
            missing = set(baseline) - set(cases)
            raise ValueError(f"{run}: case identity mismatch, missing {sorted(missing)[:5]}")
        deltas = [cases[key] - baseline[key] for key in cases]
        layers = sorted({key[0] for key in cases})
        splits = sorted({key[2] for key in cases})
        entry = {
            "overall": describe(deltas),
            "raw_gain": describe(list(cases.values())),
            "per_layer_mean": {
                str(layer): statistics.fmean([cases[key] - baseline[key] for key in cases if key[0] == layer])
                for layer in layers
            },
            "per_split_mean": {
                split: statistics.fmean([cases[key] - baseline[key] for key in cases if key[2] == split])
                for split in splits
            },
            "l1_total": statistics.fmean([abs(d) for d in deltas]),
            "l1_negative": statistics.fmean([max(-d, 0.0) for d in deltas]),
        }
        summary["arms"][run] = entry

    # Standardized residual reductions vs the strong controls (same panel).
    r_arm = 1.0 - statistics.fmean(arms["a2-gate"].values())
    for control in ("a2-strong-v168", "a2-strong-v189"):
        r_ctrl = 1.0 - statistics.fmean(arms[control].values())
        summary["arms"]["a2-gate"].setdefault("d_strong", {})[control] = (
            (r_ctrl - r_arm) / r_ctrl if r_ctrl > 0 else None
        )

    out = Path(__file__).resolve().parent / "research" / "a2_four_arm_summary.json"
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"cases per arm: {summary['case_count']}  baseline mean gain {summary['baseline_gain_mean']:.6f}")
    for run, entry in summary["arms"].items():
        ov = entry["overall"]
        print(
            f"{run:16s} mean {ov['mean']:+.6f}  median {ov['median']:+.6f}  "
            f"+/0/- {ov['positive']}/{ov['zero']}/{ov['negative']}  "
            f"L1_neg {entry['l1_negative']:.6f}  val {entry['per_split_mean'].get('validation', float('nan')):+.6f}  "
            f"test {entry['per_split_mean'].get('test', float('nan')):+.6f}"
        )
        if "d_strong" in entry:
            print(f"{'':16s} D_strong: {entry['d_strong']}")
    print(f"saved {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
