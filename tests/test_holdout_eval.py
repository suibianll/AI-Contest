from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "evaluator"))

from holdout_eval import (  # noqa: E402
    HOLDOUT_BUDGET,
    HOLDOUT_CONFIG,
    HOLDOUT_TEXT,
    check_run_allowed,
    freeze,
    holdout_seed_hash,
    load_ledger,
)
from real_data_eval import TEXT  # noqa: E402


def test_holdout_text_is_new_and_disjoint() -> None:
    dev_sentences = {s.strip() for s in TEXT.split(". ") if s.strip()}
    holdout_sentences = {
        s.strip() for s in HOLDOUT_TEXT.split(". ") if s.strip()
    }
    assert HOLDOUT_TEXT != TEXT
    # No shared sentence: the holdout text was never used for development.
    assert not (dev_sentences & holdout_sentences)
    assert len(holdout_sentences) >= 15


def test_frozen_config_has_at_least_four_token_windows() -> None:
    assert HOLDOUT_CONFIG["test"] >= 4
    assert HOLDOUT_CONFIG["calib"] >= 1
    assert HOLDOUT_CONFIG["seq"] > 0


def test_seed_hash_is_deterministic() -> None:
    assert holdout_seed_hash() == holdout_seed_hash()
    assert len(holdout_seed_hash()) == 64


def test_freeze_writes_ledger_once(tmp_path: Path) -> None:
    ledger_path = tmp_path / "holdout_ledger.json"
    freeze(ledger_path)
    data = json.loads(ledger_path.read_text(encoding="utf-8"))
    assert data["seed_hash"] == holdout_seed_hash()
    assert data["runs"] == []
    assert data["budget"] == HOLDOUT_BUDGET
    # A second freeze is a no-op, never an overwrite.
    freeze(ledger_path)
    again = json.loads(ledger_path.read_text(encoding="utf-8"))
    assert again == data


def test_check_run_allowed_enforces_once_per_candidate() -> None:
    ledger = {"seed_hash": holdout_seed_hash(), "runs": []}
    sha = "a" * 64
    check_run_allowed(ledger, sha)  # fresh candidate: allowed
    ledger["runs"].append({"solution_sha256": sha})
    with pytest.raises(RuntimeError, match="single final-holdout run"):
        check_run_allowed(ledger, sha)
    check_run_allowed(ledger, "b" * 64)  # different candidate: allowed


def test_check_run_allowed_enforces_total_budget() -> None:
    ledger = {
        "seed_hash": holdout_seed_hash(),
        "runs": [
            {"solution_sha256": chr(ord("a") + i) * 64}
            for i in range(HOLDOUT_BUDGET)
        ],
    }
    with pytest.raises(RuntimeError, match="budget exhausted"):
        check_run_allowed(ledger, "z" * 64)


def test_load_ledger_rejects_modified_holdout(tmp_path: Path) -> None:
    ledger_path = tmp_path / "holdout_ledger.json"
    freeze(ledger_path)
    data = json.loads(ledger_path.read_text(encoding="utf-8"))
    data["seed_hash"] = "0" * 64
    ledger_path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(RuntimeError, match="seed_hash"):
        load_ledger(ledger_path)
