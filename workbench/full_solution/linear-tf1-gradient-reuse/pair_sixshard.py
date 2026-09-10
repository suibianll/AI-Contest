"""Recompute the paired six-shard statistic from two single-side eval runs.

The ordinary path runs the baseline and the candidate inside one ``eval.py``
process so that ``analyzer.analyze`` can pair them directly.  That process holds
the dense pack *and* both sides' calibration states at once, and on this machine
it stopped early: ``shards_requested [0..5]``, ``stopped_early True``, two
shards written.  Shards 0 and 1 did complete and are kept as the in-process,
same-invocation evidence; they are the stronger form because the two arms were
timed inside one process.

For the remaining shards the two sides are evaluated in **separate processes**
-- each peaks at the pack plus one side's states -- and the pairing is done here
instead.  The quantity is identical: ``analyzer`` reports ``delta_mean`` over
``candidate.gain - baseline.gain`` keyed by case, and ``case_scores.linear``
carries ``case_id``, ``layer``, ``role`` and ``gain`` for every case.

The baseline side is reused from the archived ``linear-em3-v230base`` run rather
than re-measured, because it is the same root: both files record
``source_sha256 = 0f1af6db...``, which is what the script checks.  That baseline
is the one v231, v232 and now this card are all measured against, so the deltas
stay on one basis.  The same-basis check in the execution log compares the
archived baseline's shard 0 against this card's in-process shard-0 baseline.

Both sides are declared on the command line, because the two sides do not have
to come from the same invocation: the candidate run writes ``candidate-*.json``
and a same-invocation baseline would too, but a side reused from an earlier
archived run keeps whatever prefix it was written with (``--baseline-prefix``).
The two runs' recorded ``source_sha256`` are what tie them together -- the
script refuses to pair two files that disagree, and refuses to pair a side with
itself.

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
OUT_OF_SCOPE_ROLES = ("proj",)

# Default pairing: the L-TF1 candidate (two-process run) against the archived
# v230 root baseline.  Both record source_sha256 as 0f1af6db... / 0ec89710...
# respectively, and the script refuses to pair them otherwise.
DEFAULT_CAND_DIR = PROXY_V3 / "linear-tf1-cand-only" / "candidate"
DEFAULT_BASE_DIR = PROXY_V3 / "linear-em3-v230base" / "candidate"


def _read(directory: Path, prefix: str, shard: int) -> dict:
    payload = json.loads(
        (directory / f"{prefix}-linear-shard{shard}.json").read_text(encoding="utf-8")
    )
    result = payload["results"][0]
    if result.get("status") != "ok":
        raise RuntimeError(f"{directory.name} shard {shard}: status {result.get('status')!r}")
    return result


def _cases(directory: Path, prefix: str, shard: int) -> tuple[str, dict[int, dict]]:
    """Returns the result's recorded ``source_sha256`` alongside per-case scores.

    The recorded sha is what makes a cross-run pairing safe; without it there is
    nothing binding the two files to the same root.
    """
    result = _read(directory, prefix, shard)
    scores = result["case_scores"]["linear"]
    return str(result["source_sha256"]), {int(case["case_id"]): case for case in scores}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--cand-dir", type=Path, default=DEFAULT_CAND_DIR)
    parser.add_argument("--cand-prefix", default="candidate")
    parser.add_argument("--base-dir", type=Path, default=DEFAULT_BASE_DIR)
    parser.add_argument("--base-prefix", default="baseline")
    parser.add_argument("--out", type=Path, default=HERE / "paired_sixshard.json")
    parser.add_argument("--shards", default=",".join(str(s) for s in SHARDS))
    args = parser.parse_args(argv)
    shards = tuple(int(part) for part in args.shards.split(",") if part.strip())

    report: dict = {
        "protocol": "proxy-v3",
        "scenario": "linear",
        "candidate_dir": str(args.cand_dir),
        "candidate_prefix": args.cand_prefix,
        "baseline_dir": str(args.base_dir),
        "baseline_prefix": args.base_prefix,
        "shards": [],
    }
    shard_means: list[float] = []
    positive = negative = zero = 0
    total = 0
    candidate_sha: str | None = None
    baseline_sha: str | None = None
    for shard in shards:
        shard_candidate_sha, candidate = _cases(args.cand_dir, args.cand_prefix, shard)
        shard_baseline_sha, baseline = _cases(args.base_dir, args.base_prefix, shard)
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
        per_role: dict[str, list[float]] = {}
        for case_id in shared:
            case = candidate[case_id]
            delta = float(case["gain"]) - float(baseline[case_id]["gain"])
            deltas.append(delta)
            per_role.setdefault(str(case["role"]), []).append(delta)
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
            "positive_cases": pos,
            "negative_cases": neg,
            "zero_cases": zer,
            "delta_by_role": {
                role: statistics.mean(values) for role, values in sorted(per_role.items())
            },
            "candidate_timing": _read(args.cand_dir, args.cand_prefix, shard)["timing"],
            "baseline_timing": _read(args.base_dir, args.base_prefix, shard)["timing"],
        }
        report["shards"].append(entry)
        print(
            f"shard {shard}: n={len(deltas):<4} delta_mean={mean:+.6f} "
            f"median={statistics.median(deltas):+.6f}  {pos}/{neg}/{zer}",
            flush=True,
        )

    report["equal_shard_mean_delta_gain"] = statistics.mean(shard_means)
    report["shard_delta_gain_mean"] = shard_means
    report["positive_cases"] = positive
    report["negative_cases"] = negative
    report["zero_cases"] = zero
    report["case_count"] = total
    report["out_of_scope_roles"] = list(OUT_OF_SCOPE_ROLES)
    report["candidate_source_sha256"] = candidate_sha
    report["baseline_source_sha256"] = baseline_sha
    if candidate_sha == baseline_sha:
        raise RuntimeError("both sides report the same source sha; nothing was paired")
    report["method"] = (
        "two single-side eval.py processes (no in-process pairing); per-case "
        "delta = candidate.gain - baseline.gain keyed by case_id"
    )
    candidate_seconds = sum(
        float(entry["candidate_timing"]["api_total_seconds"]) for entry in report["shards"]
    )
    baseline_seconds = sum(
        float(entry["baseline_timing"]["api_total_seconds"]) for entry in report["shards"]
    )
    report["candidate_api_total_seconds_six_shards"] = candidate_seconds
    report["baseline_api_total_seconds_six_shards"] = baseline_seconds
    report["cross_process_delta_per_case_seconds"] = (
        candidate_seconds - baseline_seconds
    ) / total
    print(
        f"\nequal-shard mean {report['equal_shard_mean_delta_gain']:+.6f}   "
        f"{positive}/{negative}/{zero} of {total}",
        flush=True,
    )
    print(
        f"cross-process scoring delta "
        f"{report['cross_process_delta_per_case_seconds']:+.4f} s/call",
        flush=True,
    )
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"wrote {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
