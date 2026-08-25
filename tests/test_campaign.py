from __future__ import annotations

import json

import pytest

from hif4_system.campaign import Campaign, HoldoutBudgetExhausted


def test_failed_holdout_reservation_consumes_budget(tmp_path) -> None:
    campaign = Campaign.create(tmp_path, max_holdout_uses=1)

    reservation = campaign.reserve("holdout", "smoke")
    campaign.finish(reservation, status="crashed", report=None)

    with pytest.raises(HoldoutBudgetExhausted):
        campaign.reserve("holdout", "smoke")


def test_holdout_seeds_are_not_written_to_manifest(tmp_path) -> None:
    campaign = Campaign.create(tmp_path, max_holdout_uses=2)
    reservation = campaign.reserve("holdout", "smoke")
    manifest = json.loads((tmp_path / "campaign.json").read_text(encoding="utf-8"))

    assert reservation.seeds
    assert "seeds" not in manifest["runs"][-1]
    assert reservation.commitment in json.dumps(manifest)
    assert all(str(seed) not in json.dumps(manifest) for seed in reservation.seeds)
