from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping


class ConfigError(ValueError):
    """Raised when evaluation configuration violates system invariants."""


@dataclass(frozen=True)
class TierConfig:
    calibration_samples: int
    test_samples: int
    dev_seeds: int
    holdout_seeds: int

    def validate(self, name: str) -> None:
        values = (
            self.calibration_samples,
            self.test_samples,
            self.dev_seeds,
            self.holdout_seeds,
        )
        if any(value <= 0 for value in values):
            raise ConfigError(f"tier {name!r} values must be positive")


@dataclass(frozen=True)
class EvaluationConfig:
    schema_version: int
    backends: tuple[str, ...]
    tiers: Mapping[str, TierConfig]
    thresholds: Mapping[str, float]
    timeouts: Mapping[str, int]
    device_tolerance: float
    bootstrap_rounds: int

    def tier(self, name: str) -> TierConfig:
        try:
            return self.tiers[name]
        except KeyError as error:
            raise ConfigError(f"unknown tier: {name}") from error

    def validate(self) -> "EvaluationConfig":
        if self.schema_version != 1:
            raise ConfigError("schema_version must be 1")
        if self.backends != ("torch",):
            raise ConfigError("evaluation system is Torch-only")
        if set(self.tiers) != {"smoke", "standard", "soak"}:
            raise ConfigError("tiers must be smoke, standard, and soak")
        for name, tier in self.tiers.items():
            tier.validate(name)
        for name in self.tiers:
            if int(self.timeouts.get(name, 0)) <= 0:
                raise ConfigError(f"timeout for {name!r} must be positive")
        for name in (
            "max_negative_rate",
            "max_catastrophic_rate",
            "max_negative_rate_delta",
        ):
            value = float(self.thresholds[name])
            if not 0.0 <= value <= 1.0:
                raise ConfigError(f"threshold {name!r} must be in [0, 1]")
        if self.device_tolerance < 0.0:
            raise ConfigError("device_tolerance must be non-negative")
        if self.bootstrap_rounds <= 0:
            raise ConfigError("bootstrap_rounds must be positive")
        return self


def _default_config_path() -> Path:
    return Path(__file__).resolve().parents[1] / "config" / "default.json"


def load_config(path: Path | None) -> EvaluationConfig:
    source = path.resolve() if path is not None else _default_config_path()
    raw = json.loads(source.read_text(encoding="utf-8"))
    config = EvaluationConfig(
        schema_version=int(raw["schema_version"]),
        backends=tuple(str(value) for value in raw["backends"]),
        tiers={
            str(name): TierConfig(
                calibration_samples=int(values["calibration_samples"]),
                test_samples=int(values["test_samples"]),
                dev_seeds=int(values["dev_seeds"]),
                holdout_seeds=int(values["holdout_seeds"]),
            )
            for name, values in raw["tiers"].items()
        },
        thresholds={str(name): float(value) for name, value in raw["thresholds"].items()},
        timeouts={str(name): int(value) for name, value in raw["timeouts"].items()},
        device_tolerance=float(raw["device_tolerance"]),
        bootstrap_rounds=int(raw["bootstrap_rounds"]),
    )
    return config.validate()
