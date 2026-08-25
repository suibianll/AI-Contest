from __future__ import annotations

from pathlib import Path

import pytest

from hif4_system.registry import HashMismatch, PromotionRejected, Registry
from hif4_system.solution_loader import sha256_file


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_V9_SHA256 = "a6b8b858156164333d1d3ca25c6233b4845061f40a16d4cf74695ecdbb9041f7"


def _reports(sha256: str) -> dict[str, dict[str, object]]:
    return {
        name: {"status": "passed", "candidate_sha256": sha256}
        for name in ("gpu_dev", "cpu_dev", "holdout")
    }


def test_initial_champion_preserves_v9_hash(tmp_path: Path) -> None:
    registry = Registry(tmp_path)
    record = registry.initialize(ROOT / "solution.py", EXPECTED_V9_SHA256)

    assert record.sha256 == EXPECTED_V9_SHA256
    assert sha256_file(record.solution_path) == EXPECTED_V9_SHA256
    assert record.solution_path != (ROOT / "solution.py").resolve()
    assert registry.champion().id == record.id


def test_promotion_rejects_changed_candidate(tmp_path: Path) -> None:
    registry = Registry(tmp_path)
    registry.initialize(ROOT / "solution.py", EXPECTED_V9_SHA256)
    candidate_source = tmp_path / "candidate.py"
    candidate_source.write_text((ROOT / "tests" / "fixtures" / "minimal_solution.py").read_text(encoding="utf-8"), encoding="utf-8")
    candidate = registry.register_candidate(candidate_source, _reports(sha256_file(candidate_source)))

    candidate.solution_path.write_text("changed", encoding="utf-8")
    with pytest.raises(HashMismatch):
        registry.promote(candidate.id, candidate.sha256)


def test_promotion_requires_all_tracks_and_rollback_is_history_only(tmp_path: Path) -> None:
    registry = Registry(tmp_path)
    initial = registry.initialize(ROOT / "solution.py", EXPECTED_V9_SHA256)
    candidate_source = tmp_path / "candidate.py"
    candidate_source.write_text((ROOT / "tests" / "fixtures" / "minimal_solution.py").read_text(encoding="utf-8"), encoding="utf-8")
    candidate_sha = sha256_file(candidate_source)
    candidate = registry.register_candidate(candidate_source, _reports(candidate_sha))

    promoted = registry.promote(candidate.id, candidate_sha)
    assert promoted.id == candidate.id
    assert registry.champion().sha256 == candidate_sha
    assert len(registry.history()) >= 2

    rolled_back = registry.rollback()
    assert rolled_back.id == initial.id
    assert registry.champion().id == initial.id
    assert (tmp_path / "versions" / candidate.id / "solution.py").is_file()


def test_promotion_rejects_missing_or_failed_report(tmp_path: Path) -> None:
    registry = Registry(tmp_path)
    registry.initialize(ROOT / "solution.py", EXPECTED_V9_SHA256)
    candidate_source = tmp_path / "candidate.py"
    candidate_source.write_text("candidate", encoding="utf-8")
    candidate_sha = sha256_file(candidate_source)
    reports = _reports(candidate_sha)
    reports.pop("holdout")
    candidate = registry.register_candidate(candidate_source, reports)

    with pytest.raises(PromotionRejected):
        registry.promote(candidate.id, candidate_sha)
