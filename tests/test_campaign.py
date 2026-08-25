from __future__ import annotations

import json
from pathlib import Path

import pytest

from hif4_system.campaign import Campaign, HoldoutBudgetExhausted


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
