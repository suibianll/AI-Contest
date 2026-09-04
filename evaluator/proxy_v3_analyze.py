"""Diagnose proxy-v2/v3 parent-to-candidate regressions and runtime hotspots."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any, Mapping, Sequence

try:
    import official_eval as core
except ModuleNotFoundError as exc:  # pragma: no cover - package import path
    if exc.name != "official_eval":
        raise
    from . import official_eval as core


L1_LIMIT = 0.02
OOD_GAP_LIMIT = 0.01
OFFICIAL_TIME_LIMIT = 280.0
TIME_MODEL = {
    "intercept": 170.3,
    "hif4_calibration_and_quantize_weight": 0.1154,
    "hif4_calibration_attention": 0.6939,
    "hif4_dynamic_quantize_activation": 0.7344,
    "dynamic_qkv": -1.5837,
}


# Component deltas are emitted by proxy-v2 when decomposition is enabled.
# proxy-v3 intentionally keeps decomposition off in its fast path, but the
# analyzer accepts the same fields so a v2 default/effect result can be used
# as a slower forensic follow-up without changing the pairing policy.
_COMPONENT_DIRECTION: dict[str, tuple[str, str]] = {
    "w_only_gain": ("gain", "Weight-only arm"),
    "a_only_gain": ("gain", "Activation-only arm"),
    "both_gain": ("gain", "Both arm"),
    "interaction_gain": ("gain", "Linear W/A interaction"),
    "weight_operand_relative_mse": ("lower", "Weight operand MSE"),
    "activation_operand_relative_mse": ("lower", "Activation operand MSE"),
    "q_only_gain": ("gain", "Q-only arm"),
    "k_only_gain": ("gain", "K-only arm"),
    "v_only_gain": ("gain", "V-only arm"),
    "qk_only_gain": ("gain", "QK-only arm"),
    "qk_interaction_gain": ("gain", "Attention Q/K interaction"),
    "qkv_interaction_gain": ("gain", "Attention Q/K/V interaction"),
    "logit_mse": ("lower", "Attention logit MSE"),
    "probability_mse": ("lower", "Attention probability MSE"),
    "probability_kl": ("lower", "Attention probability KL"),
}


def _document(path: Path, result_name: str | None, label: str) -> Mapping[str, Any]:
    source = path.resolve()
    if not source.is_file():
        raise FileNotFoundError(f"evaluation JSON does not exist: {source}")
    document = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(document, dict) or not isinstance(document.get("results"), list):
        raise ValueError(f"evaluation JSON has no results list: {source}")
    if document.get("protocol") not in {core.PROTOCOL, "proxy-v3"}:
        raise ValueError(f"unsupported evaluation protocol {document.get('protocol')!r}: {source}")
    return core._select_eval_result(document, result_name, label)


def _case_key(side: str, item: Mapping[str, Any]) -> tuple[Any, ...]:
    if side == "linear":
        return core._linear_case_identity(item)
    return core._attention_case_identity(item)


def _paired_rows(
    baseline: Mapping[str, Any], candidate: Mapping[str, Any], side: str
) -> list[dict[str, Any]]:
    left = {
        _case_key(side, item): item
        for item in baseline.get("case_scores", {}).get(side, [])
    }
    right = {
        _case_key(side, item): item
        for item in candidate.get("case_scores", {}).get(side, [])
    }
    if set(left) != set(right):
        raise ValueError(f"{side} case identities differ")
    rows = []
    for key in sorted(left):
        parent = left[key]
        child = right[key]
        rows.append({
            "delta": float(child["gain"]) - float(parent["gain"]),
            "baseline_gain": float(parent["gain"]),
            "candidate_gain": float(child["gain"]),
            "layer": int(child.get("layer", -1)),
            "role": str(child.get("role", "")),
            "role_family": str(child.get("role_family", "")),
            "shape_bucket": str(child.get("shape_bucket", "")),
            "split": str(child.get("test_split", "")),
            "length": int(child.get("test_length", -1)),
        })
    return rows


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _tail_delta(rows: Sequence[Mapping[str, Any]]) -> float:
    if not rows:
        return 0.0
    count = max(1, len(rows) // 5)
    baseline = sorted(float(row["baseline_gain"]) for row in rows)[:count]
    candidate = sorted(float(row["candidate_gain"]) for row in rows)[:count]
    return _mean(candidate) - _mean(baseline)


def _group_hotspots(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    fields = ("role_family", "role", "layer", "shape_bucket", "split", "length")
    groups: list[dict[str, Any]] = []
    for field in fields:
        buckets: dict[str, list[float]] = {}
        for row in rows:
            value = str(row.get(field, ""))
            if value:
                buckets.setdefault(value, []).append(float(row["delta"]))
        for value, deltas in buckets.items():
            groups.append({
                "dimension": field,
                "value": value,
                "cases": len(deltas),
                "mean_delta": _mean(deltas),
                "negative_cases": sum(delta < 0 for delta in deltas),
                "min_delta": min(deltas),
            })
    return sorted(groups, key=lambda item: (item["mean_delta"], item["min_delta"]))


def _component_signal(components: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Normalize component deltas and mark the direction of a regression.

    Gain components use ``candidate - baseline`` (positive is better), while
    MSE/KL components use the same subtraction but negative is better.  The
    explicit direction prevents an analyzer user from misreading a positive
    probability-MSE delta as an improvement.
    """
    rows: list[dict[str, Any]] = []
    for name, raw_value in components.items():
        try:
            value = float(raw_value)
        except (TypeError, ValueError):
            continue
        direction, label = _COMPONENT_DIRECTION.get(name, ("gain", name))
        bad = value < 0.0 if direction == "gain" else value > 0.0
        rows.append({
            "component": str(name),
            "label": label,
            "delta": value,
            "direction": direction,
            "regressed": bad,
        })
    return sorted(rows, key=lambda item: (not item["regressed"], -abs(item["delta"])))


def side_analysis(
    baseline: Mapping[str, Any], candidate: Mapping[str, Any], side: str,
    paired_side: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    rows = _paired_rows(baseline, candidate, side)
    deltas = [float(row["delta"]) for row in rows]
    result = {
        "cases": len(rows),
        "delta_mean": _mean(deltas),
        "delta_median": float(statistics.median(deltas)) if deltas else 0.0,
        "l1": _mean([abs(value) for value in deltas]),
        "linf": max((abs(value) for value in deltas), default=0.0),
        "delta_tail": _tail_delta(rows),
        "positive_cases": sum(value > 0 for value in deltas),
        "negative_cases": sum(value < 0 for value in deltas),
        "zero_cases": sum(value == 0 for value in deltas),
        "worst_groups": _group_hotspots(rows)[:12],
        "worst_cases": sorted(rows, key=lambda item: float(item["delta"]))[:12],
    }
    components = dict((paired_side or {}).get("overall", {}).get("component_delta_mean", {}))
    result["component_delta_mean"] = components
    result["component_signal"] = _component_signal(components)
    return result


def runtime_analysis(candidate: Mapping[str, Any]) -> dict[str, Any]:
    timing = candidate.get("timing", {})
    api = {str(name): float(value) for name, value in timing.get("api_seconds", {}).items()}
    total = float(timing.get("api_total_seconds", sum(api.values())))
    ranked = sorted(
        (
            {
                "api": name,
                "seconds": seconds,
                "share": seconds / total if total else 0.0,
                "calls": int(timing.get("api_calls", {}).get(name, 0)),
            }
            for name, seconds in api.items()
        ),
        key=lambda item: item["seconds"], reverse=True,
    )
    faithful = bool(timing.get("calibration_timing_measured", True)) and not bool(
        timing.get("calibration_cache_hit", False)
    )
    scope = candidate.get("evaluation_scope", {})
    default_panel = "default-panel" in str(scope.get("kind", ""))
    predicted = None
    if faithful and default_panel:
        qkv = sum(api.get(name, 0.0) for name in (
            "hif4_dynamic_quantize_q", "hif4_dynamic_quantize_k", "hif4_dynamic_quantize_v"
        ))
        predicted = (
            TIME_MODEL["intercept"]
            + TIME_MODEL["hif4_calibration_and_quantize_weight"]
            * api.get("hif4_calibration_and_quantize_weight", 0.0)
            + TIME_MODEL["hif4_calibration_attention"]
            * api.get("hif4_calibration_attention", 0.0)
            + TIME_MODEL["hif4_dynamic_quantize_activation"]
            * api.get("hif4_dynamic_quantize_activation", 0.0)
            + TIME_MODEL["dynamic_qkv"] * qkv
        )
    return {
        "api_total_seconds": total,
        "calibration_wall_seconds": float(timing.get("calibration_wall_seconds", 0.0)),
        "calibration_api_seconds": float(timing.get("calibration_api_seconds", sum(
            api.get(name, 0.0) for name in (
                "hif4_calibration_and_quantize_weight", "hif4_calibration_attention"
            )
        ))),
        "calibration_cache_load_seconds": float(
            timing.get("calibration_cache_load_seconds", 0.0)
        ),
        "scoring_api_seconds": float(timing.get("scoring_api_seconds", sum(
            api.get(name, 0.0) for name in api
            if name not in {"hif4_calibration_and_quantize_weight", "hif4_calibration_attention"}
        ))),
        "scoring_wall_seconds": float(timing.get("scoring_wall_seconds", 0.0)),
        "ranked_apis": ranked,
        "timing_faithful": faithful,
        "default_panel": default_panel,
        "predicted_official_seconds": predicted,
        "under_280_gate": predicted < OFFICIAL_TIME_LIMIT if predicted is not None else None,
    }


def _ood_delta_gap(
    baseline: Mapping[str, Any], candidate: Mapping[str, Any],
    baseline_ood: Mapping[str, Any], candidate_ood: Mapping[str, Any], side: str,
) -> float | None:
    # OOD is a paired diagnostic too: do not subtract unrelated domain or
    # window panels just because both JSON files contain a non-empty list.
    _paired_rows(baseline_ood, candidate_ood, side)
    in_parent = float(baseline.get("score", {}).get(f"{side}_mean", 0.0))
    in_child = float(candidate.get("score", {}).get(f"{side}_mean", 0.0))
    ood_parent = float(baseline_ood.get("score", {}).get(f"{side}_mean", 0.0))
    ood_child = float(candidate_ood.get("score", {}).get(f"{side}_mean", 0.0))
    if not baseline_ood.get("case_scores", {}).get(side) or not candidate_ood.get("case_scores", {}).get(side):
        return None
    return (in_child - ood_child) - (in_parent - ood_parent)


def analyze(
    baseline: Mapping[str, Any],
    candidate: Mapping[str, Any],
    mechanism_type: str = "unknown",
    baseline_ood: Mapping[str, Any] | None = None,
    candidate_ood: Mapping[str, Any] | None = None,
    focus_linear_roles: Sequence[str] = (),
) -> dict[str, Any]:
    selectors = tuple(dict.fromkeys(
        str(value).strip() for value in focus_linear_roles if str(value).strip()
    ))
    paired = core._paired_effect_diagnostics(baseline, candidate, selectors)
    if not paired.get("enabled"):
        raise ValueError(f"invalid paired comparison: {paired.get('reason')}")
    sides = {
        side: side_analysis(baseline, candidate, side, paired.get(side))
        for side in ("linear", "attention")
        if candidate.get("case_scores", {}).get(side)
    }
    runtime = runtime_analysis(candidate)
    ood = {}
    if baseline_ood is not None and candidate_ood is not None:
        ood_pair = core._paired_effect_diagnostics(baseline_ood, candidate_ood)
        if not ood_pair.get("enabled"):
            raise ValueError(f"invalid OOD paired comparison: {ood_pair.get('reason')}")
        ood = {
            side: _ood_delta_gap(baseline, candidate, baseline_ood, candidate_ood, side)
            for side in sides
        }

    blockers = []
    warnings = []
    for side, item in sides.items():
        if item["delta_mean"] <= 0:
            blockers.append(f"{side}: delta_mean={item['delta_mean']:+.6f} <= 0")
        if item["l1"] >= L1_LIMIT:
            blockers.append(f"{side}: L1={item['l1']:.6f} >= {L1_LIMIT:.2f}")
        if item["delta_tail"] < 0:
            warnings.append(f"{side}: worst-20% tail regressed {item['delta_tail']:+.6f}")
        gap = ood.get(side)
        if gap is not None and abs(gap) > OOD_GAP_LIMIT:
            blockers.append(f"{side}: |delta(in-ood)|={abs(gap):.6f} > {OOD_GAP_LIMIT:.2f}")
        regressions = [component for component in item["component_signal"] if component["regressed"]]
        if regressions:
            names = ", ".join(component["component"] for component in regressions[:3])
            warnings.append(f"{side}: component regression in {names}")
    focus = paired.get("linear", {}).get("focus", {})
    control = paired.get("linear", {}).get("control", {})
    if selectors:
        if focus.get("case_count", 0) == 0:
            blockers.append("linear focus selectors matched no cases")
        elif float(focus.get("mean_delta_gain", 0.0)) <= 0.0:
            blockers.append(
                f"linear focus delta_mean={float(focus.get('mean_delta_gain', 0.0)):+.6f} <= 0"
            )
        if control.get("case_count", 0) and float(control.get("mean_delta_gain", 0.0)) < 0.0:
            blockers.append(
                f"linear control delta_mean={float(control.get('mean_delta_gain', 0.0)):+.6f} < 0"
            )
    if mechanism_type == "fitted":
        warnings.append("calibration-fitted mechanism: official history has high overfit risk")
    if runtime["under_280_gate"] is False:
        blockers.append(
            f"predicted official time {runtime['predicted_official_seconds']:.1f}s >= {OFFICIAL_TIME_LIMIT:.0f}s"
        )
    if runtime["predicted_official_seconds"] is None:
        warnings.append("no official-time prediction: run a fresh default panel without calibration cache")

    scope_kind = str(candidate.get("evaluation_scope", {}).get("kind", ""))
    if blockers:
        decision = "reject"
    elif "shard" in scope_kind:
        decision = "continue_next_shard"
    elif mechanism_type == "analytic" and runtime["under_280_gate"] is True:
        decision = "eligible_for_official_review"
    else:
        decision = "hold_for_ood_or_fresh_timing"

    hotspot = runtime["ranked_apis"][0] if runtime["ranked_apis"] else None
    actions = []
    if hotspot:
        api = hotspot["api"]
        if api == "hif4_calibration_and_quantize_weight":
            actions.append("profile Weight calibration candidate loops, matrix factorizations, and repeated quantization")
        elif api == "hif4_calibration_attention":
            actions.append("profile Attention calibration candidates and small-tensor Python control flow")
        elif api == "hif4_dynamic_quantize_activation":
            actions.append("move work from online Activation quantization into calibration state")
        else:
            actions.append(f"inspect online {api} for per-call search or small-kernel launch overhead")
    for side, item in sides.items():
        if item["worst_groups"]:
            group = item["worst_groups"][0]
            actions.append(
                f"inspect {side} {group['dimension']}={group['value']} first "
                f"(mean delta={group['mean_delta']:+.6f}, min delta={group['min_delta']:+.6f})"
            )
        for component in item["component_signal"]:
            if component["regressed"]:
                actions.append(
                    f"inspect {side} {component['label']} ({component['component']}) "
                    f"delta={component['delta']:+.6f}"
                )
                break
    return {
        "policy": {
            "score_prediction": "forbidden",
            "local_trend_gate": "delta_mean > 0 and L1 < 0.02",
            "ood_gate": "abs(delta(in_dist - ood)) <= 0.01",
            "official_time_gate": "fresh default prediction < 280s",
        },
        "mechanism_type": mechanism_type,
        "decision": decision,
        "blockers": blockers,
        "warnings": warnings,
        "sides": sides,
        "focus_linear_roles": list(selectors),
        "focus": focus if selectors else {"enabled": False, "reason": "no focus role/family supplied"},
        "control": control if selectors else {"enabled": False, "reason": "no focus role/family supplied"},
        "ood_delta_gap": ood,
        "runtime": runtime,
        "recommended_actions": actions,
    }


def render_markdown(analysis: Mapping[str, Any]) -> str:
    lines = [
        "# proxy-v3 diagnosis", "",
        f"**Decision: `{analysis['decision']}`**", "",
    ]
    if analysis["blockers"]:
        lines.extend(["## Blockers", ""] + [f"- {item}" for item in analysis["blockers"]] + [""])
    if analysis["warnings"]:
        lines.extend(["## Warnings", ""] + [f"- {item}" for item in analysis["warnings"]] + [""])
    lines.extend(["## Accuracy localization", ""])
    for side, item in analysis["sides"].items():
        lines.append(
            f"- {side}: delta_mean `{item['delta_mean']:+.6f}`, L1 `{item['l1']:.6f}`, "
            f"delta_tail `{item['delta_tail']:+.6f}`, +/-/0 "
            f"`{item['positive_cases']}/{item['negative_cases']}/{item['zero_cases']}`"
        )
        for group in item["worst_groups"][:4]:
            lines.append(
                f"  - {group['dimension']} `{group['value']}`: mean `{group['mean_delta']:+.6f}`, "
                f"min `{group['min_delta']:+.6f}`"
            )
        if item.get("component_signal"):
            lines.append("  - components: " + "; ".join(
                f"{component['component']}={component['delta']:+.6f}"
                + (" [regressed]" if component["regressed"] else "")
                for component in item["component_signal"]
            ))
    if analysis.get("focus_linear_roles"):
        focus = analysis.get("focus", {})
        control = analysis.get("control", {})
        lines.extend([
            "", "## Focus/control", "",
            f"- focus `{','.join(analysis['focus_linear_roles'])}`: "
            f"mean `{float(focus.get('mean_delta_gain', 0.0)):+.6f}`, "
            f"cases `{focus.get('case_count', 0)}`",
            f"- unmodified control: mean `{float(control.get('mean_delta_gain', 0.0)):+.6f}`, "
            f"cases `{control.get('case_count', 0)}`",
        ])
    if analysis.get("ood_delta_gap"):
        lines.extend(["", "## OOD gap", ""])
        for side, gap in analysis["ood_delta_gap"].items():
            if gap is not None:
                lines.append(f"- {side}: delta(in-ood) `{gap:+.6f}`")
    lines.extend(["", "## Runtime localization", ""])
    for item in analysis["runtime"]["ranked_apis"]:
        lines.append(
            f"- `{item['api']}`: {item['seconds']:.3f}s ({item['share']:.1%}), {item['calls']} calls"
        )
    runtime = analysis["runtime"]
    lines.append(
        f"- stages: calibration wall/API `{runtime['calibration_wall_seconds']:.3f}/"
        f"{runtime['calibration_api_seconds']:.3f}s`, scoring wall/API `"
        f"{runtime['scoring_wall_seconds']:.3f}/{runtime['scoring_api_seconds']:.3f}s`, "
        f"cache load `{runtime['calibration_cache_load_seconds']:.3f}s`"
    )
    predicted = analysis["runtime"]["predicted_official_seconds"]
    lines.append(
        f"- predicted official time: `{predicted:.1f}s`" if predicted is not None
        else "- predicted official time: unavailable (requires fresh default panel)"
    )
    if analysis["recommended_actions"]:
        lines.extend(["", "## Next actions", ""] + [f"- {item}" for item in analysis["recommended_actions"]])
    lines.extend(["", "> This tool never predicts an official score.", ""])
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--baseline-result")
    parser.add_argument("--candidate-result")
    parser.add_argument("--baseline-ood", type=Path)
    parser.add_argument("--candidate-ood", type=Path)
    parser.add_argument("--mechanism-type", choices=("analytic", "fitted", "unknown"), default="unknown")
    parser.add_argument(
        "--focus-linear-roles", default="",
        help="comma-separated Linear roles or role families for focus/control gating",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--report", type=Path)
    return parser


def main(args: argparse.Namespace) -> dict[str, Any]:
    baseline = _document(args.baseline, args.baseline_result, "baseline")
    candidate = _document(args.candidate, args.candidate_result, "candidate")
    baseline_ood = _document(args.baseline_ood, None, "baseline OOD") if args.baseline_ood else None
    candidate_ood = _document(args.candidate_ood, None, "candidate OOD") if args.candidate_ood else None
    if (baseline_ood is None) != (candidate_ood is None):
        raise ValueError("--baseline-ood and --candidate-ood must be supplied together")
    focus = tuple(item.strip() for item in args.focus_linear_roles.split(",") if item.strip())
    analysis = analyze(
        baseline, candidate, args.mechanism_type, baseline_ood, candidate_ood, focus
    )
    rendered = render_markdown(analysis)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(analysis, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(rendered, encoding="utf-8")
    print(rendered)
    return analysis


if __name__ == "__main__":
    main(build_parser().parse_args())
