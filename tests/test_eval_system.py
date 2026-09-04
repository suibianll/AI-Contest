from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "evaluator"))

import eval_system as system  # noqa: E402


def _case(index: int, gain: float, shard: int = 0) -> dict:
    return {
        "case_id": index,
        "layer": shard * 4 + index // 14,
        "role": system.core.ROLES[index % len(system.core.ROLES)],
        "role_family": "qkv",
        "shape_bucket": "hidden_to_hidden",
        "calibration_indices": [0, 1],
        "test_window": index % 2,
        "test_split": "validation" if index % 2 == 0 else "test",
        "test_length": 128,
        "gain": gain,
        "mse_standard": 1.0,
        "mse_player": 1.0 - gain,
        "reference_energy": 1.0,
    }


def _result(source: Path, shard: int, gain: float) -> dict:
    cases = [_case(index, gain, shard) for index in range(56)]
    return {
        "candidate": source.stem,
        "status": "ok",
        "source": str(source.resolve()),
        "source_sha256": system.core.sha256_file(source),
        "evaluation_scope": {
            "shard": shard,
            "shard_count": system.v3.SHARD_COUNT,
        },
        "diagnostic_config": {
            "evaluation_scenario": "linear",
            "ood": False,
        },
        "timing": {
            "api_seconds": {name: 0.0 for name in system.core.REQUIRED_APIS},
            "api_calls": {name: 0 for name in system.core.REQUIRED_APIS},
            "calibration_cache_hit": False,
            "calibration_wall_seconds": 0.0,
            "scoring_wall_seconds": 0.0,
        },
        "case_scores": {"linear": cases, "attention": []},
    }


def test_manifest_contains_current_scores_and_separates_cohorts() -> None:
    rows = system.official_results(existing_only=True)
    by_name = {row["name"]: row for row in rows}
    assert len(rows) >= 40
    assert by_name["v168"]["score"] == 14005
    assert by_name["v186"]["score"] == 17599
    assert by_name["v187"]["score"] == 9167
    assert by_name["v188"]["score"] == 17595
    assert {row["cohort"] for row in rows} == {"old-weight", "new-weight"}


def test_aggregate_selected_shards_is_weighted_and_identity_safe(tmp_path: Path) -> None:
    source = tmp_path / "solution.py"
    source.write_text("# test", encoding="utf-8")
    first = _result(source, 0, 0.1)
    second = _result(source, 1, 0.3)
    summary = system._aggregate_results([first, second])
    assert summary["sides"]["linear"]["cases"] == 112
    assert summary["sides"]["linear"]["mean"] == pytest.approx(0.2)
    assert not summary["case_identity_duplicates"]


def test_reuse_requires_dense_cache_identity(tmp_path: Path) -> None:
    source = tmp_path / "solution.py"
    source.write_text("# test", encoding="utf-8")
    cache_a = tmp_path / "a.pt"
    cache_b = tmp_path / "b.pt"
    cache_a.write_bytes(b"a")
    cache_b.write_bytes(b"b")
    result = _result(source, 0, 0.1)
    result["dense_cache"] = str(cache_a.resolve())
    assert system._result_matches(result, source, "linear", 0, False, cache_a)
    assert not system._result_matches(result, source, "linear", 0, False, cache_b)


def test_pairwise_audit_is_descriptive_and_detects_inversion() -> None:
    rows = [
        {"name": "v-a", "status": "ok", "official": {"score": 10, "cohort": "new-weight"}, "local": {"overall_mean": 0.1}},
        {"name": "v-b", "status": "ok", "official": {"score": 20, "cohort": "new-weight"}, "local": {"overall_mean": 0.0}},
    ]
    audit = system._pairwise_audit(rows)
    assert audit[0]["inverted_pairs"] == 1
    assert audit[0]["concordance_rate"] == 0.0
    assert audit[0]["near_zero_inverted_examples"] == []
    assert "no official-score conversion" in audit[0]["interpretation"]


def test_reasonableness_marks_cross_cohort_without_rejecting_finite_output(
    tmp_path: Path,
) -> None:
    source = tmp_path / "solution.py"
    source.write_text("# test", encoding="utf-8")
    result = _result(source, 0, 0.1)
    row = {
        "name": "v-old",
        "status": "ok",
        "official": {"score": 1, "cohort": "old-weight", "time_seconds": 1.0},
        "source_reproducibility": "confirmed",
        "checks": {"all_outputs_finite": True},
        "local": system._aggregate_results([result]),
    }
    checks = system._audit_reasonableness([row], "linear", [0], "new-weight")
    assert checks["complete_evaluations"] == 1
    assert any(item["kind"] == "official_cache_cohort_mismatch" for item in checks["issues"])
    assert not any(item["kind"] == "non_finite_output" for item in checks["issues"])


def test_single_manifest_partial_shard_does_not_claim_full_coverage(tmp_path: Path) -> None:
    source = tmp_path / "solution.py"
    source.write_text("# test", encoding="utf-8")
    result = _result(source, 0, 0.1)
    payload = system._single_manifest(
        output_dir=tmp_path,
        source=source,
        name="candidate",
        scenario="linear",
        shards=[0],
        ood=False,
        results=[result],
        baseline_source=None,
        baseline_results=[],
        stopped_early=False,
    )
    assert payload["checks"]["expected_case_coverage"] is True
    assert payload["candidate"]["local"]["sides"]["linear"]["cases"] == 56


def test_official_audit_reuses_one_pack_and_writes_report(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    cache = tmp_path / "dense.pt"
    cache.write_bytes(b"cache")
    source_a = tmp_path / "a.py"
    source_b = tmp_path / "b.py"
    source_a.write_text("a", encoding="utf-8")
    source_b.write_text("b", encoding="utf-8")
    records = [
        {"name": "v-a", "path": "a.py", "source": str(source_a), "source_exists": True, "score": 10, "time_seconds": 1.0, "status": "pass", "cohort": "new-weight"},
        {"name": "v-b", "path": "b.py", "source": str(source_b), "source_exists": True, "score": 20, "time_seconds": 1.0, "status": "pass", "cohort": "new-weight"},
    ]
    monkeypatch.setattr(system, "official_results", lambda **_: records)
    load_count = {"count": 0}
    monkeypatch.setattr(system.core, "load_pack", lambda path: load_count.__setitem__("count", load_count["count"] + 1) or object())
    monkeypatch.setattr(
        system.v3,
        "prepare_shard",
        lambda raw, shard, scenario, ood: SimpleNamespace(metadata={"shard": shard}),
    )

    def fake_evaluate(path, pack, device, cache_mode):
        return _result(Path(path), int(pack.metadata["shard"]), 0.2 if Path(path).name == "a.py" else 0.1)

    monkeypatch.setattr(system.v3, "evaluate", fake_evaluate)
    monkeypatch.setattr(system.v3, "cleanup_solution_modules", lambda: None)
    monkeypatch.setattr(system.v3, "make_output", lambda cache_path, pack, prepare_seconds, result, paired=None: {"protocol": "proxy-v3", "results": [result]})
    monkeypatch.setattr(
        system.v3,
        "write_output",
        lambda path, output, report=None: Path(path).parent.mkdir(parents=True, exist_ok=True) or Path(path).write_text(json.dumps(output), encoding="utf-8"),
    )
    args = system.build_parser().parse_args([
        "--official-audit",
        "--versions", "v-a,v-b",
        "--shards", "0",
        "--scenario", "linear",
        "--cache", str(cache),
        "--output-dir", str(tmp_path / "audit"),
    ])
    payload = system.run(args)
    assert load_count["count"] == 1
    assert [row["status"] for row in payload["records"]] == ["ok", "ok"]
    assert payload["pairwise"][0]["inverted_pairs"] == 1
    assert (tmp_path / "audit" / "audit.json").is_file()
    assert (tmp_path / "audit" / "audit.md").is_file()
