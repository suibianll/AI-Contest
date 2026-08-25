from __future__ import annotations

import dataclasses
from pathlib import Path

from hif4_system.optimizer import CandidateGenerator, DevFeedback, GeneratedCandidate


def test_dev_feedback_cannot_contain_holdout_cases() -> None:
    assert "holdout" not in {field.name for field in dataclasses.fields(DevFeedback)}


def test_generated_candidate_is_immutable_and_protocol_is_structural(tmp_path: Path) -> None:
    candidate = GeneratedCandidate(tmp_path / "candidate.py", {"method": "manual"})
    assert candidate.path == tmp_path / "candidate.py"
    assert isinstance(CandidateGenerator, type)
