"""Pair the six-shard readings of the A-CT2 run, case by case.

This card's claim is implementation equivalence with the same-parent A-GR1: the
baseline side here is v236 (the v231 root plus A-GR1), not the root.  A paired
delta of exactly zero on every case is therefore the *expected* reading, and a
non-zero one would be a defect -- so the claim has to rest on a per-case read
rather than on a mean that cancelling differences could also produce.

Both sides sit in the same run directory, written by one ``eval.py`` process.
This script re-derives ``delta = candidate.gain - baseline.gain`` keyed by
``case_id`` from the two sides' recorded ``case_scores.attention``.  The two
sides' recorded ``source_sha256`` are what tie them together: the script refuses
to pair two files that disagree, and refuses to pair a side with itself.

Because the candidate reproduces v236 exactly, the *mechanism's* change against
the formal parent (the v231 root) is inherited from v236's own archived
six-shard run rather than re-measured; that transitivity is only sound if the
per-case equality below holds, which is why this script is the gate for it.

Read-only with respect to the repo: prints and writes ``paired_sixshard.json``
next to itself (or to ``--out``).
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
PROXY_V3 = ROOT / "artifacts" / "proxy_v3"
SHARDS = (0, 1, 2, 3, 4, 5)

DEFAULT_RUN_DIR = PROXY_V3 / "attention-act2-sixshard-20260910" / "candidate"

EXPECTED_CASES = 72


def _read(directory: Path, prefix: str, shard: int) -> dict:
    payload = json.loads(
        (directory / f"{prefix}-attention-shard{shard}.json").read_text(encoding="utf-8")
    )
    result = payload["results"][0]
    if result.get("status") != "ok":
        raise RuntimeError(f"{directory.name} shard {shard}: status {result.get('status')!r}")
    return result


def _cases(directory: Path, prefix: str, shard: int) -> tuple[str, dict[int, dict]]:
    result = _read(directory, prefix, shard)
    scores = result["case_scores"]["attention"]
    return str(result["source_sha256"]), {int(case["case_id"]): case for case in scores}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--cand-prefix", default="candidate")
    parser.add_argument("--base-prefix", default="baseline")
    parser.add_argument("--out", type=Path, default=HERE / "paired_sixshard.json")
    parser.add_argument("--shards", default=",".join(str(s) for s in SHARDS))
    args = parser.parse_args(argv)
    shards = tuple(int(part) for part in args.shards.split(",") if part.strip())

    report: dict = {
        "protocol": "proxy-v3",
        "scenario": "attention",
        "run_dir": str(args.run_dir),
        "candidate_prefix": args.cand_prefix,
        "baseline_prefix": args.base_prefix,
        "shards": [],
    }
    shard_means: list[float] = []
    positive = negative = zero = 0
    total = 0
    candidate_sha: str | None = None
    baseline_sha: str | None = None
    for shard in shards:
        shard_candidate_sha, candidate = _cases(args.run_dir, args.cand_prefix, shard)
        shard_baseline_sha, baseline = _cases(args.run_dir, args.base_prefix, shard)
        candidate_sha = candidate_sha or shard_candidate_sha
        baseline_sha = baseline_sha or shard_baseline_sha
        if shard_candidate_sha != candidate_sha or shard_baseline_sha != baseline_sha:
            raise RuntimeError(f"shard {shard}: source sha differs from the earlier shards")
        shared = sorted(set(candidate) & set(baseline))
        if len(shared) != len(candidate) or len(shared) != len(baseline):
            raise RuntimeError(
                f"shard {shard}: case sets differ "
                f"(cand {len(candidate)}, base {len(baseline)}, shared {len(shared)})"
            )
        deltas = []
        per_layer: dict[str, list[float]] = {}
        for case_id in shared:
            case = candidate[case_id]
            delta = float(case["gain"]) - float(baseline[case_id]["gain"])
            deltas.append(delta)
            per_layer.setdefault(str(case["layer"]), []).append(delta)
        mean = statistics.mean(deltas)
        shard_means.append(mean)
        pos = sum(1 for value in deltas if value > 0)
        neg = sum(1 for value in deltas if value < 0)
        zer = sum(1 for value in deltas if value == 0)
        positive += pos
        negative += neg
        zero += zer
        total += len(deltas)
        entry = {
            "shard": shard,
            "cases": len(deltas),
            "delta_mean": mean,
            "delta_median": statistics.median(deltas),
            "delta_min": min(deltas),
            "delta_max": max(deltas),
            "positive_cases": pos,
            "negative_cases": neg,
            "zero_cases": zer,
            "delta_by_layer": {
                layer: statistics.mean(values) for layer, values in sorted(per_layer.items())
            },
            "candidate_timing": _read(args.run_dir, args.cand_prefix, shard)["timing"],
            "baseline_timing": _read(args.run_dir, args.base_prefix, shard)["timing"],
        }
        report["shards"].append(entry)
        print(
            f"shard {shard}: n={len(deltas):<3} delta_mean={mean:+.6f} "
            f"median={statistics.median(deltas):+.6f}  {pos}/{neg}/{zer}",
            flush=True,
        )

    report["equal_shard_mean_delta_gain"] = statistics.mean(shard_means)
    report["shard_delta_gain_mean"] = shard_means
    report["positive_cases"] = positive
    report["negative_cases"] = negative
    report["zero_cases"] = zero
    report["case_count"] = total
    report["candidate_source_sha256"] = candidate_sha
    report["baseline_source_sha256"] = baseline_sha
    if candidate_sha == baseline_sha:
        raise RuntimeError("both sides report the same source sha; nothing was paired")
    if total != EXPECTED_CASES:
        raise RuntimeError(f"paired {total} cases, expected {EXPECTED_CASES}")
    report["method"] = (
        "one eval.py process carrying both sides; per-case delta re-derived here as "
        "candidate.gain - baseline.gain keyed by case_id"
    )
    candidate_seconds = sum(
        float(entry["candidate_timing"]["api_total_seconds"]) for entry in report["shards"]
    )
    baseline_seconds = sum(
        float(entry["baseline_timing"]["api_total_seconds"]) for entry in report["shards"]
    )
    report["candidate_api_total_seconds_six_shards"] = candidate_seconds
    report["baseline_api_total_seconds_six_shards"] = baseline_seconds
    report["delta_api_total_seconds_six_shards"] = candidate_seconds - baseline_seconds
    calibration_candidate = sum(
        float(entry["candidate_timing"].get("hif4_calibration_attention_seconds", 0.0))
        for entry in report["shards"]
    )
    calibration_baseline = sum(
        float(entry["baseline_timing"].get("hif4_calibration_attention_seconds", 0.0))
        for entry in report["shards"]
    )
    report["calibration_attention_seconds"] = {
        "candidate": calibration_candidate,
        "baseline": calibration_baseline,
        "delta": calibration_candidate - calibration_baseline,
        "note": (
            "calibration seconds are cache-identity dependent; a cold candidate and a warm "
            "baseline are not comparable and the delta is reported as a diagnostic only"
        ),
    }
    print(
        f"\nequal-shard mean {report['equal_shard_mean_delta_gain']:+.6f}   "
        f"{positive}/{negative}/{zero} of {total}",
        flush=True,
    )
    print(
        f"attention-calibration seconds: candidate {calibration_candidate:.3f}s "
        f"baseline {calibration_baseline:.3f}s delta {calibration_candidate - calibration_baseline:+.3f}s "
        "(cache-mixed, diagnostic only)",
        flush=True,
    )
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"wrote {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
