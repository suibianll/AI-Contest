from __future__ import annotations

from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from typing import Any, Mapping


@dataclass(frozen=True)
class CaseResult:
    seed_id: str
    kind: str
    scenario: str
    test_index: int
    causal: bool
    compute_dtype: str
    mse_std: float
    mse_player: float
    score: float


@dataclass(frozen=True)
class TimingResult:
    player_quant_seconds: float
    wall_seconds: float
    peak_rss_bytes: int | None = None


@dataclass(frozen=True)
class RunResult:
    cases: tuple[CaseResult, ...]
    timing: TimingResult
    metadata: Mapping[str, Any]


def to_jsonable(value: object) -> object:
    """Convert result models and their members to JSON-supported values."""
    if is_dataclass(value) and not isinstance(value, type):
        return to_jsonable(asdict(value))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [to_jsonable(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"unsupported JSON value: {type(value).__name__}")
