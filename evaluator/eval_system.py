"""Canonical local HiF4 evaluator.

This is the user-facing replacement for the old one-shot evaluator.  It uses
the proxy-v3 shard runner by default, keeps one dense input pack per sweep,
and writes machine-readable evidence for every shard.  ``official_eval.py``
is intentionally imported as a compatibility/reference backend; it is not
mutated and its proxy-v2 JSONs are never silently treated as proxy-v3 runs.

The ``--official-audit`` mode re-evaluates every version with a recorded
official score (or an explicit subset) under the same v3 protocol.  It reports
within-cohort pairwise ordering as a diagnostic only.  It never fits or
predicts an official score from a local gain.
"""

from __future__ import annotations

import argparse
import gc
import itertools
import json
import math
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import torch

try:
    import official_eval as core
    import proxy_v3_analyze as analyzer
    import proxy_v3_eval as v3
    from official_results_v3 import official_results, manifest_summary
except ModuleNotFoundError as exc:  # pragma: no cover - package import path
    if exc.name not in {
        "official_eval", "proxy_v3_analyze", "proxy_v3_eval", "official_results_v3"
    }:
        raise
    from . import official_eval as core
    from . import proxy_v3_analyze as analyzer
    from . import proxy_v3_eval as v3
    from .official_results_v3 import official_results, manifest_summary


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = ROOT / "artifacts" / "proxy_v3" / "system"
DEFAULT_AUDIT_DIR = ROOT / "artifacts" / "proxy_v3" / "official-audit"
OFFICIAL_AUDIT_PROTOCOL = "eval-v3-official-audit"
# The checked-in dense pack is the current Qwen/WikiText capture used by the
# new-weight official runs.  Old-weight scores remain useful historical
# observations, but running their source against this pack is explicitly
# marked as a cohort mismatch in the audit.
DEFAULT_CACHE_COHORT = "new-weight"
NEAR_ZERO_LOCAL_DELTA = 0.002


def _parse_shards(value: str) -> list[int]:
    shards = [int(item.strip()) for item in value.split(",") if item.strip()]
    if (
        not shards
        or len(set(shards)) != len(shards)
        or any(item < 0 or item >= v3.SHARD_COUNT for item in shards)
    ):
        raise ValueError(
            f"shards must be comma-separated values in [0, {v3.SHARD_COUNT - 1}]"
        )
    return shards


def _result_path(directory: Path, kind: str, scenario: str, shard: int, ood: bool = False) -> Path:
    prefix = f"{kind}-ood-" if ood else f"{kind}-"
    return directory / f"{prefix}{scenario}-shard{shard}.json"


def _load_v3_result(path: Path) -> dict[str, Any]:
    document = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(document, Mapping) or document.get("protocol") != v3.PROTOCOL:
        raise ValueError(f"not a proxy-v3 document: {path}")
    result = core._select_eval_result(document, None, str(path))
    selected = dict(result)
    # ``dense_cache`` is a document-level identity in the v3 schema, while
    # ``_select_eval_result`` intentionally returns only the inner result.
    # Carry it through so --reuse-existing cannot mix two cache profiles.
    selected["dense_cache"] = document.get("dense_cache")
    return selected


def _result_matches(
    result: Mapping[str, Any],
    source: Path,
    scenario: str,
    shard: int,
    ood: bool,
    dense_cache: Path,
) -> bool:
    if result.get("status") != "ok":
        return False
    if Path(str(result.get("source", ""))).resolve() != source.resolve():
        return False
    if result.get("source_sha256") != core.sha256_file(source):
        return False
    if Path(str(result.get("dense_cache", ""))).resolve() != dense_cache.resolve():
        return False
    scope = result.get("evaluation_scope", {})
    diagnostic = result.get("diagnostic_config", {})
    return (
        int(scope.get("shard", -1)) == shard
        and int(scope.get("shard_count", -1)) == v3.SHARD_COUNT
        and str(diagnostic.get("evaluation_scenario", "")) == scenario
        and bool(diagnostic.get("ood", False)) == bool(ood)
    )


def _load_reusable_result(
    path: Path,
    source: Path,
    scenario: str,
    shard: int,
    ood: bool,
    dense_cache: Path,
) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        result = _load_v3_result(path)
    except (OSError, ValueError, KeyError, TypeError):
        return None
    return (
        result
        if _result_matches(result, source, scenario, shard, ood, dense_cache)
        else None
    )


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _aggregate_results(results: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    sides: dict[str, dict[str, Any]] = {}
    all_case_keys: dict[str, set[tuple[Any, ...]]] = {"linear": set(), "attention": set()}
    duplicate_cases: list[tuple[str, tuple[Any, ...]]] = []
    api_seconds = {name: 0.0 for name in core.REQUIRED_APIS}
    api_calls = {name: 0 for name in core.REQUIRED_APIS}
    cache_hits = 0
    calibration_walls: list[float] = []
    scoring_walls: list[float] = []
    for result in results:
        for side in ("linear", "attention"):
            cases = list(result.get("case_scores", {}).get(side, []))
            bucket = sides.setdefault(side, {"cases": 0, "gain_sum": 0.0, "gains": []})
            for item in cases:
                if side == "linear":
                    key = core._linear_case_identity(item)
                else:
                    key = core._attention_case_identity(item)
                if key in all_case_keys[side]:
                    duplicate_cases.append((side, key))
                all_case_keys[side].add(key)
                gain = float(item["gain"])
                if not _finite(gain):
                    raise ValueError("non-finite gain in proxy-v3 result")
                bucket["cases"] += 1
                bucket["gain_sum"] += gain
                bucket["gains"].append(gain)
        timing = result.get("timing", {})
        for name in core.REQUIRED_APIS:
            api_seconds[name] += float(timing.get("api_seconds", {}).get(name, 0.0))
            api_calls[name] += int(timing.get("api_calls", {}).get(name, 0))
        cache_hits += int(bool(timing.get("calibration_cache_hit", False)))
        calibration_walls.append(float(timing.get("calibration_wall_seconds", 0.0)))
        scoring_walls.append(float(timing.get("scoring_wall_seconds", 0.0)))

    side_summary: dict[str, Any] = {}
    for side, bucket in sides.items():
        gains = bucket["gains"]
        side_summary[side] = {
            "cases": bucket["cases"],
            "mean": bucket["gain_sum"] / bucket["cases"] if bucket["cases"] else 0.0,
            "median": float(statistics.median(gains)) if gains else 0.0,
            "min": min(gains) if gains else 0.0,
            "max": max(gains) if gains else 0.0,
        }
    total_cases = sum(item["cases"] for item in side_summary.values())
    total_gain = sum(item["mean"] * item["cases"] for item in side_summary.values())
    return {
        "shards": len(results),
        "sides": side_summary,
        "overall_mean": total_gain / total_cases if total_cases else 0.0,
        "case_identity_duplicates": [
            {"side": side, "key": list(key)} for side, key in duplicate_cases
        ],
        "api_seconds": api_seconds,
        "api_calls": api_calls,
        "api_total_seconds": sum(api_seconds.values()),
        "calibration_cache_hits": cache_hits,
        "calibration_wall_seconds": sum(calibration_walls),
        "scoring_wall_seconds": sum(scoring_walls),
        "calibration_timing_measured": cache_hits == 0,
    }


def _expected_cases(scenario: str) -> dict[str, int]:
    if scenario == "linear":
        return {"linear": 24 * len(core.ROLES) * 2, "attention": 0}
    if scenario == "attention":
        return {"linear": 0, "attention": 24 * 2}
    return {"linear": 24 * len(core.ROLES) * 2, "attention": 24 * 2}


def _expected_cases_for_shards(scenario: str, shard_count: int) -> dict[str, int]:
    """Expected case count for a selected shard subset."""
    per_shard = {
        "linear": 4 * len(core.ROLES) * 2,
        "attention": 4 * 2,
    }
    if scenario == "linear":
        return {"linear": per_shard["linear"] * shard_count, "attention": 0}
    if scenario == "attention":
        return {"linear": 0, "attention": per_shard["attention"] * shard_count}
    return {
        "linear": per_shard["linear"] * shard_count,
        "attention": per_shard["attention"] * shard_count,
    }


def _single_manifest(
    *,
    output_dir: Path,
    source: Path,
    name: str,
    scenario: str,
    shards: Sequence[int],
    ood: bool,
    results: Sequence[Mapping[str, Any]],
    baseline_source: Path | None,
    baseline_results: Sequence[Mapping[str, Any]],
    stopped_early: bool,
) -> dict[str, Any]:
    candidate_summary = _aggregate_results(results)
    expected = _expected_cases_for_shards(scenario, len(shards))
    actual = {side: candidate_summary["sides"].get(side, {}).get("cases", 0) for side in expected}
    checks = {
        "source_exists": source.is_file(),
        "source_sha256": core.sha256_file(source) if source.is_file() else None,
        "all_outputs_finite": all(
            _finite(item.get("gain"))
            for result in results
            for side in ("linear", "attention")
            for item in result.get("case_scores", {}).get(side, [])
        ),
        "unique_case_identities": not bool(candidate_summary["case_identity_duplicates"]),
        "expected_case_coverage": actual == expected if not stopped_early else None,
        "official_score_equivalent": False,
    }
    payload: dict[str, Any] = {
        "protocol": "eval-v3",
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "evaluator": "evaluator/eval_system.py",
        "base_protocol": v3.PROTOCOL,
        "scenario": scenario,
        "ood": ood,
        "shards_requested": list(shards),
        "stopped_early": stopped_early,
        "candidate": {
            "name": name,
            "source": str(source.resolve()),
            "source_sha256": checks["source_sha256"],
            "local": candidate_summary,
        },
        "checks": checks,
        "official_score_policy": "local proxy is diagnostic only; no absolute-score conversion",
        "results": [dict(result) for result in results],
    }
    if baseline_source is not None:
        baseline_summary = _aggregate_results(baseline_results)
        payload["baseline"] = {
            "source": str(baseline_source.resolve()),
            "source_sha256": core.sha256_file(baseline_source) if baseline_source.is_file() else None,
            "local": baseline_summary,
        }
    _write_json(output_dir / "manifest.json", payload)
    return payload


def _render_single_markdown(payload: Mapping[str, Any]) -> str:
    candidate = payload["candidate"]
    local = candidate["local"]
    lines = [
        "# eval-v3 run",
        "",
        f"- candidate: `{candidate['name']}`",
        f"- source SHA256: `{candidate.get('source_sha256')}`",
        f"- scenario/OOD: `{payload['scenario']}` / `{payload['ood']}`",
        f"- shards: `{payload['shards_requested']}`; stopped early: `{payload['stopped_early']}`",
        f"- overall local gain: `{local['overall_mean']:+.6f}`",
        f"- API total (diagnostic): `{local['api_total_seconds']:.3f}s`; calibration cache hits: `{local['calibration_cache_hits']}`",
        "- official score/time equivalent: `false`",
        "",
        "| side | cases | mean | median | min | max |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for side in ("linear", "attention"):
        item = local["sides"].get(side)
        if item:
            lines.append(
                f"| {side} | {item['cases']} | {item['mean']:+.6f} | {item['median']:+.6f} | "
                f"{item['min']:+.6f} | {item['max']:+.6f} |"
            )
    lines.extend(["", "## Checks", ""])
    for key, value in payload["checks"].items():
        lines.append(f"- {key}: `{value}`")
    if "baseline" in payload:
        lines.extend(["", "## Baseline", "", f"- source: `{payload['baseline']['source']}`"])
        for side in ("linear", "attention"):
            item = payload["baseline"]["local"]["sides"].get(side)
            if item:
                lines.append(f"- {side} mean: `{item['mean']:+.6f}` ({item['cases']} cases)")
    return "\n".join(lines) + "\n"


def _run_single(args: argparse.Namespace) -> dict[str, Any]:
    if args.solution is None:
        raise ValueError("--solution is required unless --official-audit is used")
    source = args.solution.resolve()
    if not source.is_file():
        raise FileNotFoundError(f"solution does not exist: {source}")
    scenario = args.scenario
    shards = _parse_shards(args.shards)
    output_dir = (args.output_dir / args.name).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_path = (args.cache or (core.OOD_CACHE if args.ood else core.DEFAULT_CACHE)).resolve()
    if not cache_path.is_file():
        raise FileNotFoundError(f"dense cache does not exist: {cache_path}")
    baseline_source = args.baseline_solution.resolve() if args.baseline_solution else None
    if baseline_source is not None and not baseline_source.is_file():
        raise FileNotFoundError(f"baseline solution does not exist: {baseline_source}")

    planned = []
    for shard in shards:
        planned.append(_result_path(output_dir, "candidate", scenario, shard, args.ood))
        if baseline_source is not None:
            planned.append(_result_path(output_dir, "baseline", scenario, shard, args.ood))
    raw = None
    if not args.reuse_existing or any(not path.is_file() for path in planned):
        raw = core.load_pack(cache_path)
    candidate_results: list[Mapping[str, Any]] = []
    baseline_results: list[Mapping[str, Any]] = []
    consecutive_nonpositive = 0
    stopped_early = False
    for shard in shards:
        candidate_path = _result_path(output_dir, "candidate", scenario, shard, args.ood)
        baseline_path = _result_path(output_dir, "baseline", scenario, shard, args.ood)
        candidate_result = (
            _load_reusable_result(candidate_path, source, scenario, shard, args.ood, cache_path)
            if args.reuse_existing else None
        )
        baseline_result = (
            _load_reusable_result(
                baseline_path, baseline_source, scenario, shard, args.ood, cache_path
            )
            if baseline_source is not None and args.reuse_existing else None
        )
        if candidate_result is None or (baseline_source is not None and baseline_result is None):
            if raw is None:
                raise RuntimeError("missing dense pack for a non-reusable shard")
            pack = v3.prepare_shard(raw, shard, scenario, args.ood)
            if baseline_source is not None and baseline_result is None:
                baseline_result = v3.evaluate(
                    baseline_source, pack, args.algorithm_device, args.calibration_cache_mode
                )
                v3.write_output(
                    baseline_path,
                    v3.make_output(cache_path, pack, 0.0, baseline_result),
                    baseline_path.with_suffix(".md"),
                )
                v3.cleanup_solution_modules()
            candidate_result = v3.evaluate(
                source, pack, args.algorithm_device, args.calibration_cache_mode
            )
            paired = None
            if baseline_result is not None:
                paired = core._paired_effect_diagnostics(
                    baseline_result, candidate_result, core._focus_selectors(args.focus_linear_roles)
                )
            v3.write_output(
                candidate_path,
                v3.make_output(cache_path, pack, 0.0, candidate_result, paired),
                candidate_path.with_suffix(".md"),
            )
            v3.cleanup_solution_modules()
            del pack
        candidate_results.append(candidate_result)
        if baseline_result is not None:
            baseline_results.append(baseline_result)
        if baseline_source is not None:
            paired_report = analyzer.analyze(
                baseline_result,
                candidate_result,
                args.mechanism_type,
                focus_linear_roles=core._focus_selectors(args.focus_linear_roles),
            )
            analysis_path = output_dir / f"analysis-{scenario}-shard{shard}.json"
            _write_json(analysis_path, paired_report)
            analysis_path.with_suffix(".md").write_text(
                analyzer.render_markdown(paired_report), encoding="utf-8"
            )
            side_names = [
                side for side in ("linear", "attention") if paired_report["sides"].get(side, {}).get("cases", 0)
            ]
            if any(float(paired_report["sides"][side]["delta_mean"]) <= 0.0 for side in side_names):
                consecutive_nonpositive += 1
            else:
                consecutive_nonpositive = 0
            if consecutive_nonpositive >= args.stop_after_nonpositive:
                stopped_early = True
                break

    if raw is not None:
        del raw
        gc.collect()
        v3.cleanup_solution_modules()
    payload = _single_manifest(
        output_dir=output_dir,
        source=source,
        name=args.name,
        scenario=scenario,
        shards=shards,
        ood=args.ood,
        results=candidate_results,
        baseline_source=baseline_source,
        baseline_results=baseline_results,
        stopped_early=stopped_early,
    )
    (output_dir / "manifest.md").write_text(_render_single_markdown(payload), encoding="utf-8")
    return payload


def _pairwise_audit(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    by_cohort: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        if row.get("status") == "ok" and row.get("local") is not None and row.get("official", {}).get("score") is not None:
            by_cohort.setdefault(str(row.get("official", {}).get("cohort")), []).append(row)
    audits = []
    for cohort, values in sorted(by_cohort.items()):
        concordant = inverted = tied = 0
        examples: list[dict[str, Any]] = []
        near_zero_inversions: list[dict[str, Any]] = []
        near_zero_inverted_count = 0
        for left, right in itertools.combinations(values, 2):
            local_delta = float(left["local"]["overall_mean"]) - float(right["local"]["overall_mean"])
            official_delta = float(left["official"]["score"]) - float(right["official"]["score"])
            if local_delta == 0.0 or official_delta == 0.0:
                tied += 1
                relation = "tied"
            elif local_delta * official_delta > 0.0:
                concordant += 1
                relation = "concordant"
            else:
                inverted += 1
                relation = "inverted"
            if relation == "inverted" and len(examples) < 12:
                examples.append({
                    "left": left["name"],
                    "right": right["name"],
                    "local_delta": local_delta,
                    "official_delta": official_delta,
                })
            if (
                relation == "inverted"
                and abs(local_delta) < NEAR_ZERO_LOCAL_DELTA
            ):
                near_zero_inverted_count += 1
                if len(near_zero_inversions) < 12:
                    near_zero_inversions.append({
                        "left": left["name"],
                        "right": right["name"],
                        "local_delta": local_delta,
                        "official_delta": official_delta,
                    })
        total = concordant + inverted
        audits.append({
            "cohort": cohort,
            "versions": [row["name"] for row in values],
            "concordant_pairs": concordant,
            "inverted_pairs": inverted,
            "tied_pairs": tied,
            "non_tied_pairs": total,
            "concordance_rate": concordant / total if total else None,
            "inverted_examples": examples,
            "near_zero_local_delta_threshold": NEAR_ZERO_LOCAL_DELTA,
            "near_zero_inverted_pairs": near_zero_inverted_count,
            "near_zero_inverted_examples": near_zero_inversions,
            "interpretation": "descriptive same-cohort ordering only; no official-score conversion",
        })
    return audits


def _audit_reasonableness(
    rows: Sequence[Mapping[str, Any]], scenario: str, shards: Sequence[int], cache_cohort: str
) -> dict[str, Any]:
    expected = _expected_cases_for_shards(scenario, len(shards))
    issues: list[dict[str, Any]] = []
    for row in rows:
        if row.get("status") != "ok":
            issues.append({"name": row["name"], "kind": "evaluation_error", "detail": row.get("error")})
            continue
        local = row.get("local", {})
        actual = {side: local.get("sides", {}).get(side, {}).get("cases", 0) for side in expected}
        if actual != expected:
            issues.append({"name": row["name"], "kind": "incomplete_case_coverage", "actual": actual, "expected": expected})
        if local.get("case_identity_duplicates"):
            issues.append({"name": row["name"], "kind": "duplicate_case_identity"})
        if not row.get("checks", {}).get("all_outputs_finite", False):
            issues.append({"name": row["name"], "kind": "non_finite_output"})
        official = row.get("official", {})
        if official.get("cohort") != cache_cohort:
            issues.append({
                "name": row["name"],
                "kind": "official_cache_cohort_mismatch",
                "official_cohort": official.get("cohort"),
                "cache_cohort": cache_cohort,
            })
        official_time = official.get("time_seconds")
        if official_time is not None and float(official_time) >= core.OFFICIAL_RUNTIME_LIMIT:
            issues.append({"name": row["name"], "kind": "official_time_at_or_over_limit", "seconds": official_time})
        if row.get("source_reproducibility") == "unconfirmed":
            issues.append({"name": row["name"], "kind": "source_sha_unconfirmed"})
    return {
        "expected_cases": expected,
        "requested_shards": list(shards),
        "full_shard_coverage": len(shards) == v3.SHARD_COUNT,
        "cache_cohort": cache_cohort,
        "complete_evaluations": sum(row.get("status") == "ok" for row in rows),
        "issues": issues,
        "official_score_equivalent": False,
        "absolute_mapping_used": False,
    }


def _render_audit_markdown(payload: Mapping[str, Any]) -> str:
    lines = [
        "# eval-v3 official-score audit",
        "",
        f"- protocol: `{payload['protocol']}`",
        f"- scenario: `{payload['scenario']}`; shards: `{payload['shards']}`; cache cohort: `{payload['cache_cohort']}`",
        f"- records: `{len(payload['records'])}`; official-score equivalent: `false`",
        "",
        "## Re-evaluated versions",
        "",
        "| version | cohort | official | local overall | linear | attention | status |",
        "| --- | --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in payload["records"]:
        official = row.get("official", {})
        local = row.get("local") or {}
        sides = local.get("sides", {})
        lines.append(
            f"| {row['name']} | {official.get('cohort', '')} | {official.get('score', 'NA')} | "
            f"{local.get('overall_mean', float('nan')):+.6f} | "
            f"{sides.get('linear', {}).get('mean', float('nan')):+.6f} | "
            f"{sides.get('attention', {}).get('mean', float('nan')):+.6f} | {row.get('status')} |"
        )
    lines.extend(["", "## Pairwise trend diagnostics", ""])
    for item in payload["pairwise"]:
        lines.append(
            f"- `{item['cohort']}`: concordant `{item['concordant_pairs']}`, inverted `" \
            f"{item['inverted_pairs']}`, tied `{item['tied_pairs']}`; rate `{item['concordance_rate']}`"
        )
        lines.append(
            f"  - near-zero local inversions (`|local Δ| < {item['near_zero_local_delta_threshold']}`): "
            f"`{item['near_zero_inverted_pairs']}` pairs (showing up to 12; diagnostic only)"
        )
    lines.extend(["", "## Reasonableness checks", ""])
    checks = payload["reasonableness"]
    lines.append(f"- complete evaluations: `{checks['complete_evaluations']}`")
    lines.append(f"- issues: `{len(checks['issues'])}`")
    for issue in checks["issues"]:
        lines.append(f"  - `{issue}`")
    lines.extend([
        "",
        "Local gain is retained as a diagnostic signal only. Pairwise ordering and all absolute-score comparisons are descriptive; no regression is declared from local numbers alone.",
    ])
    return "\n".join(lines) + "\n"


def _run_official_audit(args: argparse.Namespace) -> dict[str, Any]:
    if args.ood:
        raise ValueError("--official-audit is in-distribution only; run OOD as a separate candidate audit")
    shards = _parse_shards(args.shards)
    names = [item.strip() for item in args.versions.split(",") if item.strip()] if args.versions else None
    records = official_results(names=names, cohort=args.cohort, existing_only=False)
    if args.max_versions is not None:
        if args.max_versions < 1:
            raise ValueError("--max-versions must be positive")
        records = records[: args.max_versions]
    if not records:
        raise ValueError("no official-score records match the requested filter")
    output_dir = (
        DEFAULT_AUDIT_DIR
        if args.output_dir.resolve() == DEFAULT_OUTPUT_DIR.resolve()
        else args.output_dir
    ).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_path = (args.cache or core.DEFAULT_CACHE).resolve()
    if not cache_path.is_file():
        raise FileNotFoundError(f"dense cache does not exist: {cache_path}")

    source_by_name = {item["name"]: Path(item["source"]) for item in records}
    result_by_name: dict[str, list[Mapping[str, Any]]] = {item["name"]: [] for item in records}
    missing: list[tuple[dict[str, Any], int, Path]] = []
    for item in records:
        version_dir = output_dir / item["name"]
        for shard in shards:
            path = _result_path(version_dir, "candidate", args.scenario, shard)
            result = (
                _load_reusable_result(
                    path,
                    source_by_name[item["name"]],
                    args.scenario,
                    shard,
                    False,
                    cache_path,
                )
                if args.reuse_existing else None
            )
            if result is None:
                missing.append((item, shard, path))
            else:
                result_by_name[item["name"]].append(result)

    raw = core.load_pack(cache_path) if missing else None
    for shard in shards:
        shard_missing = [entry for entry in missing if entry[1] == shard]
        if not shard_missing:
            continue
        print(
            f"[official-audit] shard {shard}: {len(shard_missing)} version(s) to evaluate",
            flush=True,
        )
        pack = v3.prepare_shard(raw, shard, args.scenario, False)
        for item, _, path in shard_missing:
            source = source_by_name[item["name"]]
            print(f"[official-audit] {item['name']} shard {shard}", flush=True)
            try:
                result = v3.evaluate(source, pack, args.algorithm_device, args.calibration_cache_mode)
                result["candidate"] = item["name"]
                v3.write_output(
                    path,
                    v3.make_output(cache_path, pack, 0.0, result),
                    path.with_suffix(".md"),
                )
                result_by_name[item["name"]].append(result)
            except Exception as exc:
                result_by_name[item["name"]].append({
                    "candidate": item["name"],
                    "status": "error",
                    "source": str(source),
                    "error": f"{type(exc).__name__}: {exc}",
                })
            finally:
                v3.cleanup_solution_modules()
            print(f"[official-audit] {item['name']} shard {shard} done", flush=True)
        del pack
    if raw is not None:
        del raw
        gc.collect()
        v3.cleanup_solution_modules()

    audit_rows: list[dict[str, Any]] = []
    for item in records:
        values = result_by_name[item["name"]]
        valid = [value for value in values if value.get("status") == "ok"]
        row: dict[str, Any] = {
            "name": item["name"],
            "official": {
                "score": item.get("score"),
                "time_seconds": item.get("time_seconds"),
                "status": item.get("status"),
                "cohort": item.get("cohort"),
            },
            "source": item.get("source"),
            "source_exists": item.get("source_exists"),
            "source_reproducibility": item.get("source_reproducibility", "confirmed"),
            "status": "ok" if len(valid) == len(shards) else "error",
        }
        if valid:
            local = _aggregate_results(valid)
            row["local"] = local
            row["checks"] = {
                "all_outputs_finite": all(
                    _finite(case.get("gain"))
                    for result in valid
                    for side in ("linear", "attention")
                    for case in result.get("case_scores", {}).get(side, [])
                ),
                "unique_case_identities": not bool(local["case_identity_duplicates"]),
                "shards_complete": len(valid) == len(shards),
                "official_score_equivalent": False,
            }
        else:
            row["local"] = None
            row["checks"] = {"all_outputs_finite": False, "shards_complete": False, "official_score_equivalent": False}
            row["error"] = [value.get("error") for value in values if value.get("status") != "ok"]
        audit_rows.append(row)

    payload: dict[str, Any] = {
        "protocol": OFFICIAL_AUDIT_PROTOCOL,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "evaluator": "evaluator/eval_system.py",
        "base_protocol": v3.PROTOCOL,
        "scenario": args.scenario,
        "shards": list(shards),
        "dense_cache": str(cache_path),
        "cache_cohort": args.cache_cohort,
        "manifest": manifest_summary(records),
        "records": audit_rows,
        "pairwise": _pairwise_audit(audit_rows),
        "reasonableness": _audit_reasonableness(
            audit_rows, args.scenario, shards, args.cache_cohort
        ),
        "official_score_policy": "official scores are observations for pairwise audit only; no local-to-official mapping",
    }
    output_path = (args.audit_output or (output_dir / "audit.json")).resolve()
    _write_json(output_path, payload)
    output_path.with_suffix(".md").write_text(_render_audit_markdown(payload), encoding="utf-8")
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--official-audit", action="store_true", help="re-evaluate versions with recorded official scores")
    parser.add_argument("--solution", type=Path, help="candidate solution.py")
    parser.add_argument("--baseline-solution", type=Path, help="optional parent solution.py for paired shard diagnostics")
    parser.add_argument("--name", default="candidate")
    scenario = parser.add_mutually_exclusive_group()
    scenario.add_argument("--scenario", choices=("both", "linear", "attention"))
    scenario.add_argument("--linear-only", dest="scenario", action="store_const", const="linear")
    scenario.add_argument("--attention-only", dest="scenario", action="store_const", const="attention")
    parser.set_defaults(scenario="both")
    parser.add_argument("--shards", default=",".join(str(item) for item in range(v3.SHARD_COUNT)))
    parser.add_argument("--ood", action="store_true")
    parser.add_argument("--cache", type=Path)
    parser.add_argument("--algorithm-device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--calibration-cache-mode", choices=("off", "auto", "read", "write"), default="auto")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--reuse-existing", action="store_true", help="reuse only source/SHA/protocol-matching v3 shard JSONs")
    parser.add_argument("--focus-linear-roles", default="")
    parser.add_argument("--mechanism-type", default="analytic")
    parser.add_argument("--stop-after-nonpositive", type=int, default=2)
    parser.add_argument("--versions", default="", help="comma-separated official version names for --official-audit")
    parser.add_argument("--cohort", choices=("old-weight", "new-weight"))
    parser.add_argument(
        "--cache-cohort",
        choices=("old-weight", "new-weight"),
        default=DEFAULT_CACHE_COHORT,
        help="cohort represented by --cache; mismatches are reported, never silently merged",
    )
    parser.add_argument("--max-versions", type=int)
    parser.add_argument("--audit-output", type=Path)
    return parser


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.official_audit:
        if args.solution is not None or args.baseline_solution is not None:
            raise ValueError("--official-audit cannot be combined with --solution/--baseline-solution")
        return _run_official_audit(args)
    return _run_single(args)


if __name__ == "__main__":
    result = run(build_parser().parse_args())
    print(json.dumps({
        "protocol": result.get("protocol"),
        "output": result.get("evaluator", "evaluator/eval_system.py"),
        "records": len(result.get("records", result.get("results", []))),
        "reasonableness_issues": len(result.get("reasonableness", {}).get("issues", [])),
    }, ensure_ascii=False))
