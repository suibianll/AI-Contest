"""Fit and apply a frozen mapping from local real-model metrics to official scores.

This module calibrates the evaluator, never the candidate algorithm.  Official
scores and fitted coefficients are read only after candidate evaluation has
finished, and are never passed to ``solution.py`` or any HiF4 calibration API.

Typical usage::

    python evaluator/official_score_calibration.py fit \
      --input artifacts/real_model_suite/20260828_full.json \
      --output artifacts/real_model_suite/official_score_calibration_v0.json

    python evaluator/official_score_calibration.py predict \
      --calibration artifacts/real_model_suite/official_score_calibration_v0.json \
      --input artifacts/real_model_suite/active.json \
      --output artifacts/real_model_suite/active.prediction.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path
from typing import Any, Callable, Sequence


CALIBRATION_SCHEMA_VERSION = 1
FEATURE_GETTERS: dict[str, Callable[[dict[str, Any]], float]] = {
    "linear_macro_gain": lambda item: float(item["linear"]["macro_gain"]),
    "component_macro_gain": lambda item: float(item["linear_component_macro_gain"]),
    "linear_global_gain": lambda item: float(item["linear"]["global_gain"]),
}


class CalibrationError(RuntimeError):
    """Raised when a calibration input is incomplete or incompatible."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CalibrationError(f"cannot read JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise CalibrationError(f"JSON root must be an object: {path}")
    return value


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _mean(values: Sequence[float]) -> float:
    if not values:
        raise CalibrationError("cannot average an empty sequence")
    return sum(values) / len(values)


def _pearson(xs: Sequence[float], ys: Sequence[float]) -> float:
    if len(xs) != len(ys) or len(xs) < 2:
        raise CalibrationError("Pearson correlation requires at least two pairs")
    x_mean = _mean(xs)
    y_mean = _mean(ys)
    numerator = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys))
    denominator = math.sqrt(
        sum((x - x_mean) ** 2 for x in xs)
        * sum((y - y_mean) ** 2 for y in ys)
    )
    return numerator / denominator if denominator else 0.0


def _average_ranks(values: Sequence[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda index: values[index])
    ranks = [0.0] * len(values)
    cursor = 0
    while cursor < len(order):
        end = cursor + 1
        while end < len(order) and values[order[end]] == values[order[cursor]]:
            end += 1
        rank = (cursor + end - 1) / 2.0 + 1.0
        for index in order[cursor:end]:
            ranks[index] = rank
        cursor = end
    return ranks


def _spearman(xs: Sequence[float], ys: Sequence[float]) -> float:
    return _pearson(_average_ranks(xs), _average_ranks(ys))


def _pairwise_rank_agreement(xs: Sequence[float], ys: Sequence[float]) -> float:
    total = 0
    agreed = 0
    for left in range(len(xs)):
        for right in range(left + 1, len(xs)):
            dx = xs[left] - xs[right]
            dy = ys[left] - ys[right]
            if dx == 0 or dy == 0:
                continue
            total += 1
            agreed += int((dx > 0) == (dy > 0))
    return agreed / total if total else 0.0


def _ols(xs: Sequence[float], ys: Sequence[float]) -> dict[str, float]:
    if len(xs) != len(ys) or len(xs) < 2:
        raise CalibrationError("OLS requires at least two pairs")
    x_mean = _mean(xs)
    y_mean = _mean(ys)
    denominator = sum((x - x_mean) ** 2 for x in xs)
    if denominator <= 0:
        raise CalibrationError("local feature has zero variance across anchors")
    slope = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys)) / denominator
    intercept = y_mean - slope * x_mean
    residual_sum = sum(
        (y - (intercept + slope * x)) ** 2 for x, y in zip(xs, ys)
    )
    total_sum = sum((y - y_mean) ** 2 for y in ys)
    return {
        "intercept": intercept,
        "slope": slope,
        "r2": 1.0 - residual_sum / total_sum if total_sum else 0.0,
    }


def evaluator_contract(payload: dict[str, Any]) -> dict[str, Any]:
    run = payload.get("run")
    statuses = payload.get("model_status")
    if not isinstance(run, dict) or not isinstance(statuses, list):
        raise CalibrationError("evaluator JSON is missing run/model_status")
    loaded = [
        item for item in statuses
        if isinstance(item, dict) and item.get("status") == "loaded"
    ]
    if not loaded:
        raise CalibrationError("evaluator JSON contains no loaded models")
    model_names = [str(item["model"]) for item in loaded]
    model_revisions: dict[str, str] = {}
    data_contracts: set[tuple[str, str, str]] = set()
    for item in loaded:
        metadata = item.get("metadata")
        if not isinstance(metadata, dict):
            raise CalibrationError(f"model {item.get('model')} has no metadata")
        model_revisions[str(item["model"])] = str(metadata.get("source_revision"))
        data = metadata.get("data")
        if not isinstance(data, dict):
            raise CalibrationError(f"model {item.get('model')} has no data metadata")
        data_contracts.add(
            (str(data.get("dataset")), str(data.get("config")), str(data.get("revision")))
        )
    if len(data_contracts) != 1:
        raise CalibrationError(f"models use inconsistent dataset contracts: {data_contracts}")
    dataset, dataset_config, dataset_revision = next(iter(data_contracts))
    return {
        "mode": run.get("mode"),
        "sequence_length": run.get("sequence_length"),
        "calibration_samples": run.get("calibration_samples"),
        "test_samples": run.get("test_samples"),
        "layers": run.get("layers"),
        "models": model_names,
        "model_revisions": model_revisions,
        "dataset": dataset,
        "dataset_config": dataset_config,
        "dataset_revision": dataset_revision,
    }


def extract_candidate_features(
    payload: dict[str, Any],
    feature_name: str,
    require_official: bool,
) -> dict[str, dict[str, Any]]:
    try:
        getter = FEATURE_GETTERS[feature_name]
    except KeyError as exc:
        raise CalibrationError(f"unsupported feature: {feature_name}") from exc
    contract = evaluator_contract(payload)
    expected_models = list(contract["models"])
    results = payload.get("results")
    if not isinstance(results, list):
        raise CalibrationError("evaluator JSON has no results list")

    grouped: dict[str, dict[str, Any]] = {}
    for result in results:
        if not isinstance(result, dict) or "error" in result:
            continue
        try:
            candidate = str(result["candidate"])
            model = str(result["model"])
            value = getter(result)
        except (KeyError, TypeError, ValueError) as exc:
            raise CalibrationError("malformed candidate result") from exc
        if not math.isfinite(value):
            raise CalibrationError(f"non-finite {feature_name} for {candidate}/{model}")
        entry = grouped.setdefault(
            candidate,
            {
                "model_values": {},
                "source_sha256": set(),
                "official_scores": set(),
                "official_times": set(),
            },
        )
        if model in entry["model_values"]:
            raise CalibrationError(f"duplicate result for {candidate}/{model}")
        entry["model_values"][model] = value
        source_sha = result.get("source_sha256")
        if source_sha:
            entry["source_sha256"].add(str(source_sha))
        if result.get("official_score") is not None:
            entry["official_scores"].add(int(result["official_score"]))
        if result.get("official_time") is not None:
            entry["official_times"].add(float(result["official_time"]))

    if not grouped:
        raise CalibrationError("evaluator JSON contains no successful candidate results")
    normalized: dict[str, dict[str, Any]] = {}
    for candidate, entry in grouped.items():
        actual_models = list(entry["model_values"])
        if set(actual_models) != set(expected_models):
            missing = sorted(set(expected_models) - set(actual_models))
            extra = sorted(set(actual_models) - set(expected_models))
            raise CalibrationError(
                f"candidate {candidate} does not cover the evaluator model panel; "
                f"missing={missing}, extra={extra}"
            )
        if len(entry["source_sha256"]) != 1:
            raise CalibrationError(f"candidate {candidate} has inconsistent source SHA256 values")
        if len(entry["official_scores"]) > 1 or len(entry["official_times"]) > 1:
            raise CalibrationError(f"candidate {candidate} has inconsistent official results")
        if require_official and len(entry["official_scores"]) != 1:
            raise CalibrationError(f"candidate {candidate} has no official score")
        ordered_values = [entry["model_values"][model] for model in expected_models]
        normalized[candidate] = {
            "feature_value": _mean(ordered_values),
            "model_values": {
                model: entry["model_values"][model] for model in expected_models
            },
            "source_sha256": next(iter(entry["source_sha256"])),
            "official_score": (
                next(iter(entry["official_scores"])) if entry["official_scores"] else None
            ),
            "official_time": (
                next(iter(entry["official_times"])) if entry["official_times"] else None
            ),
        }
    return normalized


def fit_calibration(
    evaluator_payload: dict[str, Any],
    feature_name: str = "linear_macro_gain",
    minimum_anchors: int = 4,
    training_input: Path | None = None,
) -> dict[str, Any]:
    anchors = extract_candidate_features(
        evaluator_payload, feature_name, require_official=True
    )
    if len(anchors) < minimum_anchors:
        raise CalibrationError(
            f"at least {minimum_anchors} official anchors are required; got {len(anchors)}"
        )
    names = list(anchors)
    xs = [float(anchors[name]["feature_value"]) for name in names]
    ys = [float(anchors[name]["official_score"]) for name in names]
    params = _ols(xs, ys)
    fitted = [params["intercept"] + params["slope"] * value for value in xs]
    residuals = [actual - prediction for actual, prediction in zip(ys, fitted)]
    loo_errors: list[float] = []
    loo_predictions: dict[str, float] = {}
    for held_out, name in enumerate(names):
        train_x = xs[:held_out] + xs[held_out + 1 :]
        train_y = ys[:held_out] + ys[held_out + 1 :]
        loo = _ols(train_x, train_y)
        prediction = loo["intercept"] + loo["slope"] * xs[held_out]
        loo_predictions[name] = prediction
        loo_errors.append(abs(prediction - ys[held_out]))
    loo_mae = _mean(loo_errors)
    residual_rmse = math.sqrt(_mean([value * value for value in residuals]))

    anchor_records = []
    for index, name in enumerate(names):
        item = anchors[name]
        anchor_records.append(
            {
                "candidate": name,
                "source_sha256": item["source_sha256"],
                "local_feature": xs[index],
                "model_values": item["model_values"],
                "official_score": int(ys[index]),
                "official_time": item["official_time"],
                "fitted_score": fitted[index],
                "residual": residuals[index],
                "leave_one_out_prediction": loo_predictions[name],
                "leave_one_out_absolute_error": loo_errors[index],
            }
        )
    training_source = None
    if training_input is not None:
        training_source = {
            "path": str(training_input),
            "sha256": sha256_file(training_input),
        }
    return {
        "schema_version": CALIBRATION_SCHEMA_VERSION,
        "kind": "official_score_calibration",
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "status": "diagnostic" if len(anchors) < 8 else "provisional",
        "feature": feature_name,
        "aggregation": "arithmetic_mean_across_exact_model_panel",
        "model": {
            "type": "ordinary_least_squares_1d",
            "intercept": params["intercept"],
            "slope": params["slope"],
            "training_feature_min": min(xs),
            "training_feature_max": max(xs),
        },
        "diagnostics": {
            "anchor_count": len(anchors),
            "pearson": _pearson(xs, ys),
            "spearman": _spearman(xs, ys),
            "pairwise_rank_agreement": _pairwise_rank_agreement(xs, ys),
            "r2": params["r2"],
            "residual_rmse_points": residual_rmse,
            "leave_one_anchor_out_mae_points": loo_mae,
            "leave_one_anchor_out_max_error_points": max(loo_errors),
        },
        "evaluator_contract": evaluator_contract(evaluator_payload),
        "training_source": training_source,
        "anchors": anchor_records,
        "warning": (
            "This maps evaluator metrics to historical official scores. It is not an "
            "official evaluator replica or a confidence guarantee, and it must never be "
            "passed into solution.py or used to infer Q(A) from A@W."
        ),
    }


def _contract_mismatches(
    expected: dict[str, Any], actual: dict[str, Any]
) -> dict[str, dict[str, Any]]:
    keys = (
        "mode",
        "sequence_length",
        "calibration_samples",
        "test_samples",
        "layers",
        "models",
        "model_revisions",
        "dataset",
        "dataset_config",
        "dataset_revision",
    )
    return {
        key: {"expected": expected.get(key), "actual": actual.get(key)}
        for key in keys
        if expected.get(key) != actual.get(key)
    }


def predict_scores(
    calibration: dict[str, Any],
    evaluator_payload: dict[str, Any],
    calibration_path: Path | None = None,
    evaluator_input: Path | None = None,
) -> dict[str, Any]:
    if calibration.get("schema_version") != CALIBRATION_SCHEMA_VERSION:
        raise CalibrationError("unsupported calibration schema")
    if calibration.get("kind") != "official_score_calibration":
        raise CalibrationError("input is not an official-score calibration")
    feature_name = str(calibration.get("feature"))
    if feature_name not in FEATURE_GETTERS:
        raise CalibrationError(f"unsupported calibration feature: {feature_name}")
    expected_contract = calibration.get("evaluator_contract")
    if not isinstance(expected_contract, dict):
        raise CalibrationError("calibration has no evaluator contract")
    actual_contract = evaluator_contract(evaluator_payload)
    mismatches = _contract_mismatches(expected_contract, actual_contract)
    if mismatches:
        raise CalibrationError(f"evaluator contract mismatch: {mismatches}")
    candidates = extract_candidate_features(
        evaluator_payload, feature_name, require_official=False
    )
    model = calibration.get("model")
    diagnostics = calibration.get("diagnostics")
    if not isinstance(model, dict) or not isinstance(diagnostics, dict):
        raise CalibrationError("calibration model/diagnostics are missing")
    intercept = float(model["intercept"])
    slope = float(model["slope"])
    feature_min = float(model["training_feature_min"])
    feature_max = float(model["training_feature_max"])
    loo_mae = float(diagnostics["leave_one_anchor_out_mae_points"])
    anchor_by_name = {
        str(item["candidate"]): item
        for item in calibration.get("anchors", [])
        if isinstance(item, dict) and "candidate" in item
    }
    baseline = anchor_by_name.get("c39")

    predictions = []
    for name, item in candidates.items():
        value = float(item["feature_value"])
        predicted = intercept + slope * value
        in_range = feature_min <= value <= feature_max
        record: dict[str, Any] = {
            "candidate": name,
            "source_sha256": item["source_sha256"],
            "feature": feature_name,
            "feature_value": value,
            "model_values": item["model_values"],
            "predicted_official_score": predicted,
            "predicted_official_score_rounded": int(round(predicted)),
            "estimated_absolute_error_points": loo_mae,
            "heuristic_two_mae_range": [
                int(round(predicted - 2.0 * loo_mae)),
                int(round(predicted + 2.0 * loo_mae)),
            ],
            "heuristic_range_is_not_a_confidence_interval": True,
            "within_training_feature_range": in_range,
            "extrapolation": not in_range,
            "actual_official_score": item["official_score"],
        }
        if item["official_score"] is not None:
            record["prediction_error"] = predicted - float(item["official_score"])
        if baseline is not None:
            baseline_local = float(baseline["local_feature"])
            baseline_official = float(baseline["official_score"])
            predicted_delta = slope * (value - baseline_local)
            record["versus_c39"] = {
                "local_feature_delta": value - baseline_local,
                "predicted_official_delta": predicted_delta,
                "c39_anchored_official_score": baseline_official + predicted_delta,
                "predicted_above_c39": predicted_delta > 0,
            }
        predictions.append(record)

    return {
        "schema_version": CALIBRATION_SCHEMA_VERSION,
        "kind": "official_score_predictions",
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "calibration": {
            "path": str(calibration_path) if calibration_path else None,
            "sha256": sha256_file(calibration_path) if calibration_path else None,
            "status": calibration.get("status"),
            "feature": feature_name,
            "anchor_count": diagnostics.get("anchor_count"),
        },
        "evaluator_input": {
            "path": str(evaluator_input) if evaluator_input else None,
            "sha256": sha256_file(evaluator_input) if evaluator_input else None,
        },
        "evaluator_contract": actual_contract,
        "predictions": predictions,
        "warning": calibration.get("warning"),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    fit = commands.add_parser("fit", help="fit a frozen calibration from official anchors")
    fit.add_argument("--input", type=Path, required=True, help="anchor evaluator JSON")
    fit.add_argument("--output", type=Path, required=True, help="calibration JSON")
    fit.add_argument(
        "--feature",
        choices=tuple(FEATURE_GETTERS),
        default="linear_macro_gain",
    )
    fit.add_argument("--minimum-anchors", type=int, default=4)

    predict = commands.add_parser("predict", help="apply a frozen calibration")
    predict.add_argument("--calibration", type=Path, required=True)
    predict.add_argument("--input", type=Path, required=True, help="candidate evaluator JSON")
    predict.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "fit":
        if args.minimum_anchors < 3:
            raise SystemExit("--minimum-anchors must be at least 3")
        calibration = fit_calibration(
            read_json(args.input),
            feature_name=args.feature,
            minimum_anchors=args.minimum_anchors,
            training_input=args.input,
        )
        write_json(args.output, calibration)
        diagnostics = calibration["diagnostics"]
        model = calibration["model"]
        print(
            f"CALIBRATION feature={calibration['feature']} anchors={diagnostics['anchor_count']} "
            f"intercept={model['intercept']:.6f} slope={model['slope']:.6f} "
            f"LOO_MAE={diagnostics['leave_one_anchor_out_mae_points']:.2f}"
        )
        print(f"JSON: {args.output}")
        return 0
    if args.command == "predict":
        predictions = predict_scores(
            read_json(args.calibration),
            read_json(args.input),
            calibration_path=args.calibration,
            evaluator_input=args.input,
        )
        write_json(args.output, predictions)
        for item in predictions["predictions"]:
            print(
                f"PREDICTION candidate={item['candidate']} "
                f"score={item['predicted_official_score']:.2f} "
                f"estimated_abs_error={item['estimated_absolute_error_points']:.2f} "
                f"extrapolation={item['extrapolation']}"
            )
        print(f"JSON: {args.output}")
        return 0
    raise AssertionError(f"unhandled command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
