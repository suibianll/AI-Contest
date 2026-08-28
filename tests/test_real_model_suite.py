from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "evaluator"))

from real_model_suite import (  # noqa: E402
    CACHE_SCHEMA_VERSION,
    CacheValidationError,
    ModelData,
    ModelSpec,
    MODEL_SPECS,
    WIKITEXT_FILES,
    WIKITEXT_CONFIG,
    WIKITEXT_REVISION,
    Window,
    _aggregate_details,
    audit_official_ranking,
    load_real_windows,
    load_model_cache,
    model_cache_path,
    save_model_cache,
    validate_window_split,
)
from real_data_eval import load_solution  # noqa: E402


def test_solution_loader_requires_only_the_six_official_apis(tmp_path: Path) -> None:
    source = tmp_path / "six_api_solution.py"
    source.write_text(
        "\n".join(
            [
                "def hif4_calibration_and_quantize_weight(*args): pass",
                "def hif4_dynamic_quantize_activation(*args): pass",
                "def hif4_calibration_attention(*args): pass",
                "def hif4_dynamic_quantize_q(*args): pass",
                "def hif4_dynamic_quantize_k(*args): pass",
                "def hif4_dynamic_quantize_v(*args): pass",
            ]
        ),
        encoding="utf-8",
    )
    loaded = load_solution(source)
    assert not hasattr(loaded, "_dequantize_hif4")


def test_official_aggregation_sums_per_case_relative_scores() -> None:
    details = [
        {
            "gain": 0.25,
            "score_sum": 0.5,
            "case_count": 2,
            "standard_sum": 100.0,
            "player_sum": 80.0,
            "elements": 8,
        },
        {
            "gain": -0.5,
            "score_sum": -0.5,
            "case_count": 1,
            "standard_sum": 1.0,
            "player_sum": 2.0,
            "elements": 4,
        },
    ]
    aggregate = _aggregate_details(details)
    assert aggregate["official_score_sum"] == 0.0
    assert aggregate["official_case_count"] == 3
    assert aggregate["official_score_mean"] == 0.0
    assert aggregate["global_gain"] != aggregate["official_score_mean"]


def test_official_ranking_audit_uses_summed_linear_and_attention_score() -> None:
    def result(candidate: str, model: str, linear: float, attention: float) -> dict:
        return {
            "candidate": candidate,
            "model": model,
            "official_flow_score": {
                "linear": linear,
                "attention": attention,
                "total": linear + attention,
            },
            "linear": {"global_gain": 0.0, "macro_gain": 0.0},
            "linear_component_macro_gain": 0.0,
            "attention": {"global_gain": 0.0, "macro_gain": 0.0},
            "timing": {"official_api_total_seconds": 10.0},
        }

    rows = [
        result("a", "m1", 3.0, -1.0),
        result("a", "m2", 2.0, 0.0),
        result("b", "m1", 1.0, 2.0),
        result("b", "m2", 0.0, 2.0),
    ]
    audit = audit_official_ranking(rows, ["a", "b"], {})
    totals = audit["aggregate_features"]["official_flow_total"]
    assert totals == {"a": 4.0, "b": 5.0}
    assert audit["candidate_status"]["a"]["valid_submission"] is True


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


def _minimal_cache_data(tmp_path: Path) -> tuple[ModelSpec, ModelData]:
    spec = ModelSpec("cache-test", "gpt2", tmp_path / "model-not-needed", "test@revision")
    roles = ("q", "k", "v", "o", "fc", "proj")
    role_groups = {role: role for role in roles}
    calibration_window = Window(
        "train", "train-doc", 0, 0, 0, 4, (1, 2, 3, 4)
    )
    test_window = Window(
        "validation", "validation-doc", 0, 0, 0, 4, (5, 6, 7, 8)
    )
    weights = [{role: torch.ones(4, 4) for role in roles}]
    activations = {
        role: [[torch.ones(4, 4)]] for role in roles
    }
    qkv = [[(torch.ones(4, 4), torch.ones(4, 4), torch.ones(4, 4))]]
    metadata = {
        "model": spec.name,
        "family": spec.family,
        "source_revision": spec.source_revision,
        "data": {
            "dataset": "Salesforce/wikitext",
            "config": WIKITEXT_CONFIG,
            "revision": WIKITEXT_REVISION,
        },
    }
    return spec, ModelData(
        spec=spec,
        tokenizer_name="test-tokenizer",
        layers=1,
        hidden_size=4,
        q_heads=2,
        kv_heads=2,
        head_dim=2,
        roles=roles,
        role_groups=role_groups,
        weights=weights,
        calibration_activations=activations,
        test_activations=activations,
        calibration_qkv=qkv,
        test_qkv=qkv,
        calibration_windows=[calibration_window],
        test_windows=[test_window],
        metadata=metadata,
    )


def test_model_cache_round_trip_is_independent_of_model_files(tmp_path: Path) -> None:
    spec, data = _minimal_cache_data(tmp_path)
    path = model_cache_path(spec, tmp_path / "cache", 4, 1, 1, None)
    assert path.name.endswith(f"schema{CACHE_SCHEMA_VERSION}.pt")
    save_model_cache(data, path, requested_layers=None)

    restored = load_model_cache(path, spec, 4, 1, 1, None)
    assert restored.metadata["loaded_from_cache"] is True
    assert restored.metadata["cache_path"] == str(path)
    assert restored.weights[0]["q"].device.type == "cpu"
    assert restored.test_qkv[0][0][0].shape == (4, 4)
    assert not spec.path.exists(), "cache loading must not require the model directory"


def test_model_cache_rejects_configuration_mismatch(tmp_path: Path) -> None:
    spec, data = _minimal_cache_data(tmp_path)
    path = model_cache_path(spec, tmp_path / "cache", 4, 1, 1, None)
    save_model_cache(data, path, requested_layers=None)
    with pytest.raises(CacheValidationError, match="configuration mismatch"):
        load_model_cache(path, spec, 8, 1, 1, None)
