"""Run proxy-v3 shards sequentially and stop after repeated clear regressions.

The runner keeps one dense proxy-v2 pack in memory and evaluates the parent and
candidate in-process.  This avoids rereading the multi-GB dense cache for every
shard; ``--reuse-existing`` avoids loading it entirely when all JSONs exist.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

try:
    import official_eval as core
except ModuleNotFoundError as exc:  # pragma: no cover - package import path
    if exc.name != "official_eval":
        raise
    from . import official_eval as core
try:
    from proxy_v3_analyze import analyze, render_markdown
    import proxy_v3_eval as v3
except ModuleNotFoundError as exc:  # pragma: no cover - package import path
    if exc.name not in {"proxy_v3_analyze", "proxy_v3_eval"}:
        raise
    from .proxy_v3_analyze import analyze, render_markdown
    from . import proxy_v3_eval as v3

SHARD_COUNT = v3.SHARD_COUNT


def _result(path: Path) -> dict[str, Any]:
    document = json.loads(path.read_text(encoding="utf-8"))
    if document.get("protocol") != v3.PROTOCOL:
        raise ValueError(f"unsupported evaluation protocol in {path}")
    return dict(core._select_eval_result(document, None, str(path)))


def _shard_output_path(
    output_dir: Path, kind: str, scenario: str, shard: int, ood: bool
) -> Path:
    prefix = f"{kind}-{scenario}-"
    if ood:
        prefix = f"{kind}-ood-{scenario}-"
    return output_dir / f"{prefix}shard{shard}.json"


def run(args: argparse.Namespace) -> dict[str, Any]:
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.stop_after_nonpositive < 1:
        raise ValueError("stop-after-nonpositive must be at least 1")
    shards = [int(value) for value in args.shards.split(",") if value.strip()]
    if (
        not shards
        or len(set(shards)) != len(shards)
        or any(value < 0 or value >= SHARD_COUNT for value in shards)
    ):
        raise ValueError(f"shards must be comma-separated values in [0, {SHARD_COUNT - 1}]")
    cache_path = (args.cache or (core.OOD_CACHE if args.ood else core.DEFAULT_CACHE)).resolve()
    planned_outputs = [
        _shard_output_path(args.output_dir, kind, args.scenario, shard, args.ood)
        for shard in shards
        for kind in ("baseline", "candidate")
    ]
    raw = None
    if not args.reuse_existing or any(not path.is_file() for path in planned_outputs):
        if not cache_path.is_file():
            raise FileNotFoundError(f"dense proxy-v2 cache does not exist: {cache_path}")
        # Load the 11-GB dense cache once for the whole sequential run.  The
        # old subprocess implementation re-read it for every baseline and
        # candidate, making a six-shard sweep I/O-bound before calibration.
        raw = core.load_pack(cache_path)
    records = []
    consecutive_nonpositive: dict[str, int] = {"linear": 0, "attention": 0}
    cumulative: dict[str, dict[str, float]] = {
        "linear": {"cases": 0, "delta": 0.0, "l1": 0.0},
        "attention": {"cases": 0, "delta": 0.0, "l1": 0.0},
    }
    for shard in shards:
        baseline_json = _shard_output_path(
            args.output_dir, "baseline", args.scenario, shard, args.ood
        )
        candidate_json = _shard_output_path(
            args.output_dir, "candidate", args.scenario, shard, args.ood
        )
        baseline_result = _result(baseline_json) if (
            args.reuse_existing and baseline_json.is_file()
        ) else None
        candidate_result = _result(candidate_json) if (
            args.reuse_existing and candidate_json.is_file()
        ) else None
        if baseline_result is None or candidate_result is None:
            if raw is None:
                raise RuntimeError("runner has no dense pack for a missing shard result")
            prepare_started = time.perf_counter()
            pack = v3.prepare_shard(raw, shard, args.scenario, args.ood)
            prepare_seconds = time.perf_counter() - prepare_started
            if baseline_result is None:
                baseline_result = v3.evaluate(
                    args.baseline_solution.resolve(), pack, args.algorithm_device,
                    args.calibration_cache_mode,
                )
                v3.write_output(
                    baseline_json,
                    v3.make_output(cache_path, pack, prepare_seconds, baseline_result),
                    baseline_json.with_suffix(".md"),
                )
                v3.cleanup_solution_modules()
            if candidate_result is None:
                candidate_result = v3.evaluate(
                    args.candidate_solution.resolve(), pack, args.algorithm_device,
                    args.calibration_cache_mode,
                )
                paired = core._paired_effect_diagnostics(
                    baseline_result,
                    candidate_result,
                    core._focus_selectors(args.focus_linear_roles),
                )
                v3.write_output(
                    candidate_json,
                    v3.make_output(cache_path, pack, prepare_seconds, candidate_result, paired),
                    candidate_json.with_suffix(".md"),
                )
                v3.cleanup_solution_modules()
            del pack
        report = analyze(
            baseline_result, candidate_result, args.mechanism_type,
            focus_linear_roles=tuple(
                item.strip() for item in args.focus_linear_roles.split(",") if item.strip()
            ),
        )
        present_sides = [
            side for side in ("linear", "attention")
            if report["sides"].get(side, {}).get("cases", 0)
        ]
        if not present_sides:
            raise RuntimeError(f"shard {shard} produced no cases")
        analysis_json = _shard_output_path(
            args.output_dir, "analysis", args.scenario, shard, args.ood
        )
        analysis_report = analysis_json.with_suffix(".md")
        analysis_json.write_text(
            json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        analysis_report.write_text(render_markdown(report), encoding="utf-8")
        side_metrics: dict[str, dict[str, float | int]] = {}
        for side in present_sides:
            item = report["sides"][side]
            delta = float(item["delta_mean"])
            side_cases = int(item["cases"])
            cumulative[side]["cases"] += side_cases
            cumulative[side]["delta"] += delta * side_cases
            cumulative[side]["l1"] += float(item["l1"]) * side_cases
            consecutive_nonpositive[side] = (
                consecutive_nonpositive[side] + 1 if delta <= 0.0 else 0
            )
            side_metrics[side] = {
                "cases": side_cases,
                "delta_mean": delta,
                "l1": float(item["l1"]),
                "delta_tail": float(item["delta_tail"]),
            }
        aggregate_cases = sum(int(cumulative[side]["cases"]) for side in present_sides)
        aggregate_delta = sum(cumulative[side]["delta"] for side in present_sides)
        aggregate_l1 = sum(cumulative[side]["l1"] for side in present_sides)
        # Keep one compact top-level metric for shell users.  For a combined
        # run it is an unweighted-by-official-score case mean; the per-side
        # metrics remain authoritative in ``side_metrics`` and ``aggregate``.
        if args.scenario == "both":
            side = "both"
            delta = sum(
                float(report["sides"][name]["delta_mean"])
                * int(report["sides"][name]["cases"])
                for name in present_sides
            ) / sum(int(report["sides"][name]["cases"]) for name in present_sides)
            side_cases = sum(int(report["sides"][name]["cases"]) for name in present_sides)
            l1 = sum(
                float(report["sides"][name]["l1"])
                * int(report["sides"][name]["cases"])
                for name in present_sides
            ) / side_cases
            tail = min(float(report["sides"][name]["delta_tail"]) for name in present_sides)
        else:
            side = present_sides[0]
            side_item = report["sides"][side]
            delta = float(side_item["delta_mean"])
            side_cases = int(side_item["cases"])
            l1 = float(side_item["l1"])
            tail = float(side_item["delta_tail"])
        records.append({
            "shard": shard,
            "baseline": str(baseline_json.resolve()),
            "candidate": str(candidate_json.resolve()),
            "decision": report["decision"],
            "side": side,
            "delta_mean": delta,
            "l1": l1,
            "delta_tail": tail,
            "cases": side_cases,
            "cumulative_delta_mean": aggregate_delta / aggregate_cases,
            "cumulative_l1": aggregate_l1 / aggregate_cases,
            "side_metrics": side_metrics,
            "analysis_json": str(analysis_json.resolve()),
            "analysis_report": str(analysis_report.resolve()),
            "blockers": report["blockers"],
            "recommended_actions": report["recommended_actions"],
        })
        if any(
            consecutive_nonpositive[side] >= args.stop_after_nonpositive
            for side in present_sides
        ):
            break

    if raw is not None:
        del raw
        v3.cleanup_solution_modules()
    stopped_early = len(records) < len(shards)
    manifest = {
        "protocol": "proxy-v3",
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "baseline_solution": str(args.baseline_solution.resolve()),
        "candidate_solution": str(args.candidate_solution.resolve()),
        "scenario": args.scenario,
        "ood": args.ood,
        "dense_cache": str(cache_path),
        "algorithm_device": args.algorithm_device,
        "calibration_cache_mode": args.calibration_cache_mode,
        "mechanism_type": args.mechanism_type,
        "focus_linear_roles": [
            item.strip() for item in args.focus_linear_roles.split(",") if item.strip()
        ],
        "requested_shards": shards,
        "completed_shards": [item["shard"] for item in records],
        "stopped_early": stopped_early,
        "stop_rule": f"per side: {args.stop_after_nonpositive} consecutive shards with delta_mean <= 0; any side stops",
        "records": records,
        "aggregate": {
            "cases": sum(int(item["cases"]) for item in cumulative.values()),
            "delta_mean": sum(item["delta"] for item in cumulative.values())
            / max(1, sum(int(item["cases"]) for item in cumulative.values())),
            "l1": sum(item["l1"] for item in cumulative.values())
            / max(1, sum(int(item["cases"]) for item in cumulative.values())),
            "by_side": {
                side: {
                    "cases": int(item["cases"]),
                    "delta_mean": item["delta"] / item["cases"] if item["cases"] else 0.0,
                    "l1": item["l1"] / item["cases"] if item["cases"] else 0.0,
                }
                for side, item in cumulative.items()
                if item["cases"]
            },
        },
        "official_score_prediction": None,
    }
    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    lines = [
        "# proxy-v3 sequential run", "",
        f"- scenario: `{args.scenario}`",
        f"- completed shards: `{manifest['completed_shards']}`",
        f"- stopped early: `{stopped_early}`",
        f"- aggregate delta_mean/L1: `{manifest['aggregate']['delta_mean']:+.6f}` / "
        f"`{manifest['aggregate']['l1']:.6f}` over `{manifest['aggregate']['cases']}` cases",
        "- official score prediction: `disabled`", "",
        "| shard | cases | delta_mean | cumulative mean | L1 | decision |",
        "|---:|---:|---:|---:|---:|---|",
    ]
    for item in records:
        lines.append(
            f"| {item['shard']} | {item['cases']} | {item['delta_mean']:+.6f} | "
            f"{item['cumulative_delta_mean']:+.6f} | {item['l1']:.6f} | {item['decision']} |"
        )
    (args.output_dir / "manifest.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-solution", type=Path, required=True)
    parser.add_argument("--candidate-solution", type=Path, required=True)
    parser.add_argument("--scenario", choices=("linear", "attention", "both"), required=True)
    parser.add_argument("--shards", default="0,1,2,3,4,5")
    parser.add_argument("--stop-after-nonpositive", type=int, default=2)
    parser.add_argument("--mechanism-type", choices=("analytic", "fitted", "unknown"), default="unknown")
    parser.add_argument("--focus-linear-roles", default="")
    parser.add_argument("--ood", action="store_true")
    parser.add_argument("--cache", type=Path)
    parser.add_argument(
        "--algorithm-device",
        default="cuda" if v3.torch.cuda.is_available() else "cpu",
    )
    parser.add_argument(
        "--calibration-cache-mode", choices=("off", "auto", "read", "write"), default="auto"
    )
    parser.add_argument("--reuse-existing", action="store_true")
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


if __name__ == "__main__":
    manifest = run(build_parser().parse_args())
    print(
        f"proxy-v3 completed shards={manifest['completed_shards']} "
        f"stopped_early={manifest['stopped_early']} "
        f"aggregate_delta={manifest['aggregate']['delta_mean']:+.6f}"
    )
