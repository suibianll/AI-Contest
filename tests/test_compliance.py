from __future__ import annotations

import math

import pytest
import torch

from hif4_system.compliance import check_static, clone_state, validate_state


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


def test_state_rejects_oversized_utf8_strings_keys_and_sparse_tensors() -> None:
    with pytest.raises(ValueError, match="4096 UTF-8 bytes"):
        validate_state("中" * 1366)
    with pytest.raises(ValueError, match="4096 UTF-8 bytes"):
        validate_state({"中" * 1366: 1})
    with pytest.raises(ValueError, match="dense strided"):
        validate_state(
            torch.sparse_coo_tensor([[0]], [1.0], (2,), check_invariants=False)
        )


def test_clone_state_is_independent_for_every_mutable_container_and_tensor() -> None:
    original = {"tensor": torch.tensor([1.0]), "list": [{"value": 2}]}

    cloned = clone_state(original)
    cloned["tensor"][0] = 9.0
    cloned["list"][0]["value"] = 7

    assert original["tensor"].item() == 1.0
    assert original["list"][0]["value"] == 2
