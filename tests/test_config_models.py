from __future__ import annotations

import json
from dataclasses import replace

import pytest

from hif4_system.config import ConfigError, load_config
from hif4_system.models import CaseResult, to_jsonable


def test_default_config_is_torch_only() -> None:
    config = load_config(None)

    assert set(config.tiers) == {"smoke", "standard", "soak"}
    assert config.compute_dtypes == ("fp32",)
    assert config.attention_causal_modes == (False,)
    assert config.timeouts["standard"] == 300
    assert config.backends == ("torch",)
    assert config.tier("smoke").calibration_samples == 1


def test_case_result_round_trip_is_json_safe() -> None:
    row = CaseResult(
        seed_id="seed-00",
        kind="linear",
        scenario="balanced",
        test_index=0,
        causal=False,
        compute_dtype="fp32",
        mse_std=1.0,
        mse_player=0.5,
        score=0.5,
    )

    decoded = json.loads(json.dumps(to_jsonable(row)))

    assert decoded["score"] == 0.5
    assert decoded["scenario"] == "balanced"


def test_unknown_tier_is_rejected() -> None:
    config = load_config(None)

    with pytest.raises(ConfigError, match="unknown tier"):
        config.tier("nightly")


def test_non_torch_backend_is_rejected() -> None:
    config = load_config(None)

    with pytest.raises(ConfigError, match="Torch-only"):
        replace(config, backends=("torch", "numpy")).validate()
