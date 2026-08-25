from __future__ import annotations

import math

import pytest
import torch

from hif4_system.compliance import check_static, validate_state


def test_file_io_is_rejected(tmp_path) -> None:
    candidate = tmp_path / "bad.py"
    candidate.write_text("def hif4_dynamic_quantize_q(x):\n    return open('x')\n", encoding="utf-8")

    report = check_static(candidate)

    assert "file_io" in report.violations
    assert not report.passed


def test_valid_nested_cpu_state_is_accepted() -> None:
    validate_state({"mode": "safe", "x": torch.ones(3), "flags": [True, 2]})


def test_state_rejects_nan_gpu_and_deep_values() -> None:
    with pytest.raises(ValueError, match="finite"):
        validate_state(float("nan"))
    with pytest.raises(ValueError, match="CPU"):
        validate_state(torch.ones(2, device="meta"))
    nested = value = {}
    for _ in range(9):
        nested["next"] = {}
        nested = nested["next"]
    with pytest.raises(ValueError, match="depth"):
        validate_state(value)
