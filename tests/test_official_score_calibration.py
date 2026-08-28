from __future__ import annotations

import copy
import math
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "evaluator"))

from official_score_calibration import (  # noqa: E402
    CalibrationError,
    fit_calibration,
    predict_scores,
    read_json,
)
from real_model_suite import (  # noqa: E402
    CandidateSpec,
    build_parser as build_suite_parser,
    fit_official_anchors,
)


ANCHOR_MATRIX = ROOT / "artifacts" / "real_model_suite" / "20260828_full.json"


def test_real_official_anchor_calibration_matches_recorded_fit() -> None:
    payload = read_json(ANCHOR_MATRIX)
    calibration = fit_calibration(payload, training_input=ANCHOR_MATRIX)

    assert calibration["feature"] == "linear_macro_gain"
    assert calibration["status"] == "diagnostic"
    assert calibration["diagnostics"]["anchor_count"] == 4
    assert math.isclose(
        calibration["model"]["intercept"], 10701.8883212338, rel_tol=1e-12
    )
    assert math.isclose(
        calibration["model"]["slope"], 9905.92863792173, rel_tol=1e-12
    )
    assert math.isclose(
        calibration["diagnostics"]["leave_one_anchor_out_mae_points"],
        160.897344418088,
        rel_tol=1e-12,
    )


def test_real_anchor_prediction_is_versioned_and_reports_uncertainty() -> None:
    payload = read_json(ANCHOR_MATRIX)
    calibration = fit_calibration(payload)
    output = predict_scores(calibration, payload)
    predictions = {item["candidate"]: item for item in output["predictions"]}

    assert predictions["c40"]["predicted_official_score_rounded"] == 14410
    assert predictions["c40"]["actual_official_score"] == 14432
    assert predictions["c40"]["within_training_feature_range"] is True
    assert predictions["c40"]["heuristic_range_is_not_a_confidence_interval"] is True
    assert math.isclose(
        predictions["c39"]["versus_c39"]["predicted_official_delta"], 0.0, abs_tol=1e-12
    )


def test_prediction_rejects_evaluator_contract_mismatch() -> None:
    payload = read_json(ANCHOR_MATRIX)
    calibration = fit_calibration(payload)
    incompatible = copy.deepcopy(payload)
    incompatible["run"]["sequence_length"] = 256

    with pytest.raises(CalibrationError, match="evaluator contract mismatch"):
        predict_scores(calibration, incompatible)


def test_calibration_rejects_incomplete_model_panel() -> None:
    payload = read_json(ANCHOR_MATRIX)
    incomplete = copy.deepcopy(payload)
    incomplete["results"] = [
        item
        for item in incomplete["results"]
        if not (item.get("candidate") == "c21" and item.get("model") == "gpt2-small")
    ]
    with pytest.raises(CalibrationError, match="does not cover the evaluator model panel"):
        fit_calibration(incomplete)


def test_suite_accepts_an_arbitrary_solution_source() -> None:
    args = build_suite_parser().parse_args(
        ["--solution", "solution.py", "--candidate-name", "active", "--cache-mode", "read"]
    )
    assert args.solution == Path("solution.py")
    assert args.candidate_name == "active"
    assert args.candidates is None


def test_custom_candidate_does_not_fit_against_missing_official_score() -> None:
    payload = read_json(ANCHOR_MATRIX)
    c40_results = [
        {**item, "candidate": "active", "official_score": None, "official_time": None}
        for item in payload["results"]
        if item.get("candidate") == "c40"
    ]
    specs = {"active": CandidateSpec("active", ROOT / "solution.py", None, None)}
    fit = fit_official_anchors(c40_results, ["active"], specs)
    assert fit["official_anchor_scores"] == {}
    assert fit["fit"] == {}
