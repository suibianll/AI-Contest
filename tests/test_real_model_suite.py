from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "evaluator"))

from real_model_suite import (  # noqa: E402
    MODEL_SPECS,
    WIKITEXT_FILES,
    Window,
    load_real_windows,
    validate_window_split,
)


def test_manifest_covers_distinct_model_families() -> None:
    assert {spec.family for spec in MODEL_SPECS.values()} == {
        "gpt2",
        "opt",
        "gpt_neox",
        "qwen2",
    }
    assert len(MODEL_SPECS) >= 5


def test_window_validator_rejects_calibration_test_document_leakage() -> None:
    calibration = [Window("train", "doc-a", 0, 2, 0, 128, tuple(range(128)))]
    test = [Window("validation", "doc-a", 0, 2, 128, 256, tuple(range(128)))]
    with pytest.raises(ValueError, match="source-document leakage"):
        validate_window_split(calibration, test, 128)


def test_window_validator_rejects_overlapping_ranges() -> None:
    calibration = [
        Window("train", "doc-a", 0, 2, 0, 128, tuple(range(128))),
        Window("train", "doc-a", 0, 2, 64, 192, tuple(range(128))),
    ]
    test = [Window("validation", "doc-b", 0, 2, 0, 128, tuple(range(128)))]
    with pytest.raises(ValueError, match="overlapping token windows"):
        validate_window_split(calibration, test, 128)


@pytest.mark.skipif(
    not (ROOT / "models" / "gpt2").is_dir()
    or not all(
        (ROOT / "data" / "wikitext-2-raw-v1" / filename).is_file()
        for filename in WIKITEXT_FILES.values()
    ),
    reason="real local model/corpus assets are not installed",
)
def test_real_wikitext_windows_are_split_without_reuse() -> None:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        ROOT / "models" / "gpt2", local_files_only=True
    )
    calibration, test, metadata = load_real_windows(
        tokenizer,
        ROOT / "data" / "wikitext-2-raw-v1",
        sequence_length=128,
        calibration_samples=2,
        test_samples=4,
    )
    assert len(calibration) == 2
    assert len(test) == 4
    assert not ({window.document_id for window in calibration} & {window.document_id for window in test})
    assert metadata["no_source_document_overlap"] is True
    assert metadata["no_token_window_overlap"] is True


def test_suite_does_not_use_synthetic_attention_for_ranking() -> None:
    source = (ROOT / "evaluator" / "real_model_suite.py").read_text(encoding="utf-8")
    assert "import synthetic_attention_eval" not in source
    assert "from synthetic_attention_eval" not in source
