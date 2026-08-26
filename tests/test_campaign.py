from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from hif4_system.campaign import Campaign, HoldoutBudgetExhausted
from hif4_system.config import load_config


def test_new_campaign_has_full_soak_development_schedule(tmp_path: Path) -> None:
    campaign = Campaign.create(tmp_path / "campaign")
    reservation = campaign.reserve("dev", "soak")

    assert reservation.seeds == (101, 211, 307, 401, 503, 607, 709, 809)


def test_legacy_short_schedule_is_extended_deterministically(tmp_path: Path) -> None:
    directory = tmp_path / "campaign"
    campaign = Campaign.create(directory)
    manifest_path = directory / "campaign.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["dev_seeds"] = [101, 211, 307]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    reservation = Campaign.open(directory).reserve("dev", "soak")

    assert reservation.seeds == (101, 211, 307, 401, 503, 607, 709, 809)
    stored = json.loads(manifest_path.read_text(encoding="utf-8"))["dev_seeds"]
    assert stored == [101, 211, 307, 401, 503, 607, 709, 809]


def test_failed_holdout_reservation_consumes_budget(tmp_path: Path) -> None:
    campaign = Campaign.create(tmp_path / "holdout", max_holdout_uses=1)

    reservation = campaign.reserve("holdout", "smoke")
    campaign.finish(reservation, status="crashed", report=None)

    with pytest.raises(HoldoutBudgetExhausted):
        campaign.reserve("holdout", "smoke")


def test_holdout_seeds_are_not_written_to_manifest(tmp_path: Path) -> None:
    campaign = Campaign.create(tmp_path / "holdout", max_holdout_uses=2)
    reservation = campaign.reserve("holdout", "smoke")
    manifest = json.loads((tmp_path / "holdout" / "campaign.json").read_text(encoding="utf-8"))

    assert reservation.seeds
    assert "seeds" not in manifest["runs"][-1]
    assert reservation.commitment in json.dumps(manifest)
    assert all(str(seed) not in json.dumps(manifest) for seed in reservation.seeds)



def test_same_commitment_reservations_finish_independently(tmp_path: Path) -> None:
    campaign = Campaign.create(tmp_path / "parallel")
    first = campaign.reserve("dev", "smoke")
    second = campaign.reserve("dev", "smoke")

    assert first.commitment == second.commitment
    assert first.reservation_id != second.reservation_id

    campaign.finish(first, status="passed", report="first.json")
    campaign.finish(second, status="failed", report="second.json")
    runs = json.loads(
        (tmp_path / "parallel" / "campaign.json").read_text(encoding="utf-8")
    )["runs"]

    assert runs[-2]["status"] == "passed"
    assert runs[-2]["report"] == "first.json"
    assert runs[-1]["status"] == "failed"
    assert runs[-1]["report"] == "second.json"



def test_policy_is_locked_after_first_holdout(tmp_path: Path) -> None:
    campaign = Campaign.create(tmp_path / "locked")
    config = load_config(None)
    reservation = campaign.reserve("holdout", "smoke", config)
    campaign.finish(reservation, status="passed", report="holdout.json")
    changed = replace(
        config,
        thresholds={**config.thresholds, "min_mean_score": 0.25},
    )

    with pytest.raises(ValueError, match="policy changed"):
        campaign.reserve("dev", "smoke", changed)
