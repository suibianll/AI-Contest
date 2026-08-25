"""Safe extension point for future automatic candidate generation.

Phase one intentionally exposes a protocol rather than an optimizer that can
rewrite the competition source.  A later generator receives only development
feedback and a read-only Champion path; holdout cases and seeds are absent from
the type by construction.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Protocol, Sequence


@dataclass(frozen=True)
class DevFeedback:
    champion_sha256: str
    candidate_sha256: str
    dev_summaries: Mapping[str, Mapping[str, object]]
    failure_categories: Mapping[str, tuple[str, ...]]
    remaining_dev_budget: int


@dataclass(frozen=True)
class GeneratedCandidate:
    path: Path
    metadata: Mapping[str, object]


class CandidateGenerator(Protocol):
    def generate(
        self,
        champion: Path,
        feedback: DevFeedback,
        output_dir: Path,
    ) -> Sequence[GeneratedCandidate]:
        """Return source candidates without access to holdout data."""

