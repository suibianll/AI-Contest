from __future__ import annotations

import json
import math
import sys
from types import SimpleNamespace
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "evaluator"))

import official_eval as v2  # noqa: E402
import proxy_v3_eval as v3  # noqa: E402
import proxy_v3_runner as runner  # noqa: E402
from proxy_v3_analyze import _document, analyze, runtime_analysis  # noqa: E402
from proxy_v3_eval import (  # noqa: E402
    PROTOCOL,
    evaluate,
    SHARD_COUNT,
    load_calibration_artifact,
    prepare_shard,
    save_calibration_artifact,
    shard_layers,
)


def _raw_pack() -> v2.RawPack:
    layers = 24
    roles = v2.ROLES
    calibration_windows = [
        v2.Window("train", f"cal-{index}", 0, 0, 0, length, tuple(range(length)))
        for index, length in enumerate(v2.CALIBRATION_LENGTHS)
    ]
    test_windows = [
        v2.Window(
            "validation" if index < 5 else "test",
            f"test-{index}", 0, 0, 0, length, tuple(range(length)),
        )
        for index, length in enumerate(v2.TEST_LENGTHS)
    ]
    weights = [
        {role: torch.ones(2, 64) for role in roles}
        for _ in range(layers)
    ]
    calibration_activations = {
        role: [[torch.ones(2, 64) for _ in range(layers)] for _ in calibration_windows]
        for role in roles
    }
    test_activations = {
        role: [[torch.ones(2, 64) for _ in range(layers)] for _ in test_windows]
        for role in roles
    }
    calibration_qkv = [
        [(torch.ones(2, 64), torch.ones(2, 64), torch.ones(2, 64)) for _ in range(layers)]
        for _ in calibration_windows
    ]
    test_qkv = [
        [(torch.ones(2, 64), torch.ones(2, 64), torch.ones(2, 64)) for _ in range(layers)]
        for _ in test_windows
    ]
    return v2.RawPack(
        weights, calibration_activations, test_activations,
        calibration_qkv, test_qkv, calibration_windows, test_windows,
        layers, 64, 1, 1, 64,
        {"model_revision": "test", "data_sha256": {"train": "test"}}, roles,
    )


def test_proxy_v3_shards_cover_each_state_once() -> None:
    raw = _raw_pack()
    all_linear = []
    all_attention = []
    for shard in range(SHARD_COUNT):
        pack = prepare_shard(raw, shard, "both")
        assert pack.metadata["protocol"] == PROTOCOL
        assert shard_layers(24, shard) == tuple(range(shard, 24, SHARD_COUNT))
        assert len(pack.metadata["linear_state_keys"]) == 28
        assert len(pack.metadata["attention_state_layers"]) == 4
        assert len(pack.linear_cases) == 56
        assert len(pack.attention_cases) == 8
        all_linear.extend(tuple(item) for item in pack.metadata["linear_state_keys"])
        all_attention.extend(pack.metadata["attention_state_layers"])
        for layer, role in (tuple(item) for item in pack.metadata["linear_state_keys"]):
            cases = [case for case in pack.linear_cases if case.layer == layer and case.role == role]
            assert {raw.test_windows[case.test_window].split for case in cases} == {"validation", "test"}
            assert len({len(raw.test_windows[case.test_window].input_ids) for case in cases}) == 1
    assert len(all_linear) == len(set(all_linear)) == 24 * len(v2.ROLES)
    assert len(all_attention) == len(set(all_attention)) == 24


def test_calibration_artifact_round_trip_and_identity_guard(tmp_path: Path) -> None:
    pack = prepare_shard(_raw_pack(), 0, "linear")
    params = v2.encode_standard_hif4(torch.ones(2, 64))
    states = {
        (layer, role): (None, params)
        for layer, role in (tuple(item) for item in pack.metadata["linear_state_keys"])
    }
    identity = {"key": "value"}
    path = tmp_path / "calibration.pt"
    save_calibration_artifact(path, identity, states, {})
    loaded_states, loaded_attention = load_calibration_artifact(path, identity, pack)
    assert set(loaded_states) == set(states)
    assert loaded_attention == {}
    with pytest.raises(RuntimeError, match="identity mismatch"):
        load_calibration_artifact(path, {"key": "different"}, pack)


def _result(
    name: str,
    gains: list[float],
    api_seconds: dict[str, float] | None = None,
    with_components: bool = False,
) -> dict:
    cases = []
    for index, gain in enumerate(gains):
        cases.append({
            "case_id": index,
            "layer": index // 2,
            "role": "q" if index % 2 == 0 else "fc_gate",
            "role_family": "qkv" if index % 2 == 0 else "fc",
            "shape_bucket": "hidden_to_hidden",
            "calibration_indices": [0, 1],
            "test_window": index,
            "test_split": "validation" if index % 2 == 0 else "test",
            "test_length": 128,
            "gain": gain,
            "mse_standard": 1.0,
            "mse_player": 1.0 - gain,
            "relative_player_mse": 1.0 - gain,
            "reference_energy": 1.0,
        })
        if with_components:
            cases[-1].update({
                "gain_w_only": gain,
                "gain_a_only": gain / 2.0,
                "gain_both": gain,
                "interaction_gain": 0.0,
            })
    seconds = api_seconds or {name: 0.0 for name in v2.REQUIRED_APIS}
    return {
        "candidate": name,
        "status": "ok",
        "evaluation_scope": {"kind": "linear-only-sequential-shard"},
        "score": {"linear_mean": sum(gains) / len(gains), "attention_mean": 0.0, "overall_mean": sum(gains) / len(gains)},
        "timing": {
            "api_seconds": seconds,
            "api_total_seconds": sum(seconds.values()),
            "api_calls": {name: 1 for name in seconds},
            "calibration_cache_hit": False,
            "calibration_timing_measured": True,
        },
        "case_scores": {"linear": cases, "attention": []},
    }


def test_analysis_rejects_regression_and_locates_group() -> None:
    baseline = _result("parent", [0.2, 0.2, 0.2, 0.2])
    candidate = _result("child", [0.21, 0.15, 0.21, 0.15])
    report = analyze(baseline, candidate, "analytic")
    assert report["decision"] == "reject"
    assert report["sides"]["linear"]["delta_mean"] < 0
    assert report["sides"]["linear"]["worst_groups"][0]["mean_delta"] < 0
    assert any("delta_mean" in blocker for blocker in report["blockers"])
    assert report["policy"]["score_prediction"] == "forbidden"


def test_runtime_prediction_requires_fresh_default_panel() -> None:
    seconds = {name: 0.0 for name in v2.REQUIRED_APIS}
    seconds["hif4_calibration_and_quantize_weight"] = 100.0
    result = _result("candidate", [0.1], seconds)
    assert runtime_analysis(result)["predicted_official_seconds"] is None
    result["evaluation_scope"]["kind"] = "default-panel"
    predicted = runtime_analysis(result)["predicted_official_seconds"]
    assert predicted is not None and math.isclose(predicted, 181.84)
    result["timing"]["calibration_cache_hit"] = True
    assert runtime_analysis(result)["predicted_official_seconds"] is None


def test_component_and_focus_diagnostics_are_actionable() -> None:
    baseline = _result("parent", [0.2, 0.2, 0.2, 0.2], with_components=True)
    candidate = _result("child", [0.19, 0.21, 0.19, 0.21], with_components=True)
    for item in candidate["case_scores"]["linear"]:
        item["gain_w_only"] -= 0.05
        item["gain_a_only"] += 0.01
    report = analyze(
        baseline,
        candidate,
        "analytic",
        focus_linear_roles=("q",),
    )
    assert report["focus_linear_roles"] == ["q"]
    assert report["focus"]["case_count"] == 2
    assert report["control"]["case_count"] == 2
    assert report["sides"]["linear"]["component_delta_mean"]["w_only_gain"] < 0
    assert any("component regression" in warning for warning in report["warnings"])
    assert any("linear focus" in blocker for blocker in report["blockers"])


def test_evaluate_fake_solution_exercises_both_sides(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    class StandardSolution:
        @staticmethod
        def _standard(carrier: torch.Tensor, scale: torch.Tensor) -> dict[str, torch.Tensor]:
            return v2.encode_standard_hif4(v2.dequantize_nvfp4(carrier, scale))

        def hif4_calibration_and_quantize_weight(self, carrier, scale, calibration):
            return {"weight_params": self._standard(carrier, scale), "activation_state": {}}

        def hif4_dynamic_quantize_activation(self, carrier, scale, state):
            return self._standard(carrier, scale)

        def hif4_calibration_attention(self, calibration, q_heads, kv_heads, head_dim):
            return {"q_state": {}, "k_state": {}, "v_state": {}}

        def hif4_dynamic_quantize_q(self, carrier, scale, heads, head_dim, state):
            return self._standard(carrier, scale)

        def hif4_dynamic_quantize_k(self, carrier, scale, heads, head_dim, state):
            return self._standard(carrier, scale)

        def hif4_dynamic_quantize_v(self, carrier, scale, heads, head_dim, state):
            return self._standard(carrier, scale)

    monkeypatch.setattr(v3.v2, "load_solution", lambda path: StandardSolution())
    monkeypatch.setattr(
        v3,
        "_calibration_identity",
        lambda path, pack, device: {
            "protocol": PROTOCOL,
            "solution_sha256": "x" * 64,
            "scenario": pack.metadata["evaluation_scenario"],
        },
    )
    pack = prepare_shard(_raw_pack(), 0, "both")
    artifact = tmp_path / "calibration.pt"
    result = evaluate(Path("synthetic_solution.py"), pack, "cpu", "write", artifact)
    assert result["score"]["linear_cases"] == 56
    assert result["score"]["attention_cases"] == 8
    assert result["timing"]["calibration_cache_hit"] is False
    assert result["timing"]["calibration_api_seconds"] > 0.0
    assert result["timing"]["scoring_api_seconds"] > 0.0
    cached = evaluate(Path("synthetic_solution.py"), pack, "cpu", "read", artifact)
    assert cached["timing"]["calibration_cache_hit"] is True
    assert cached["timing"]["calibration_timing_measured"] is False
    assert cached["timing"]["calibration_api_seconds"] == 0.0


def test_analyzer_accepts_v3_result_documents(tmp_path: Path) -> None:
    result = _result("candidate", [0.1])
    path = tmp_path / "v3.json"
    path.write_text(
        json.dumps({"protocol": "proxy-v3", "results": [result]}),
        encoding="utf-8",
    )
    assert _document(path, None, "candidate")["candidate"] == "candidate"


def test_runner_reuses_existing_shards_without_loading_dense_pack(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    cache = tmp_path / "dense.pt"
    monkeypatch.setattr(runner.core, "DEFAULT_CACHE", cache)
    for shard in range(SHARD_COUNT):
        baseline = _result("parent", [0.2, 0.2, 0.2, 0.2])
        candidate = _result("child", [0.21, 0.21, 0.21, 0.21])
        for name, result in (("baseline", baseline), ("candidate", candidate)):
            (tmp_path / f"{name}-linear-shard{shard}.json").write_text(
                json.dumps({"protocol": "proxy-v3", "results": [result]}),
                encoding="utf-8",
            )
    args = runner.build_parser().parse_args([
        "--baseline-solution", "parent.py",
        "--candidate-solution", "candidate.py",
        "--scenario", "linear",
        "--output-dir", str(tmp_path),
        "--reuse-existing",
    ])
    manifest = runner.run(args)
    assert manifest["completed_shards"] == list(range(SHARD_COUNT))
    assert manifest["aggregate"]["cases"] == SHARD_COUNT * 4
    assert manifest["aggregate"]["delta_mean"] == pytest.approx(0.01)


def test_runner_loads_dense_pack_once_for_multiple_shards(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    cache = tmp_path / "dense.pt"
    cache.write_bytes(b"placeholder")
    monkeypatch.setattr(runner.core, "DEFAULT_CACHE", cache)
    load_count = {"count": 0}

    def fake_load_pack(path):
        load_count["count"] += 1
        return object()

    monkeypatch.setattr(runner.core, "load_pack", fake_load_pack)
    monkeypatch.setattr(
        runner.v3,
        "prepare_shard",
        lambda raw, shard, scenario, ood: SimpleNamespace(
            metadata={"shard": shard, "evaluation_scenario": scenario, "ood": ood}
        ),
    )
    monkeypatch.setattr(
        runner.v3,
        "evaluate",
        lambda path, pack, device, cache_mode: _result(
            "parent" if "parent" in str(path) else "child", [0.2]
        ),
    )
    monkeypatch.setattr(
        runner.v3,
        "make_output",
        lambda cache_path, pack, prepare_seconds, result, paired=None: {
            "protocol": "proxy-v3", "results": [result], "paired_effect": paired
        },
    )
    monkeypatch.setattr(runner.v3, "cleanup_solution_modules", lambda: None)
    monkeypatch.setattr(
        runner.v3,
        "write_output",
        lambda path, output, report=None: path.write_text(
            json.dumps(output), encoding="utf-8"
        ),
    )
    args = runner.build_parser().parse_args([
        "--baseline-solution", "parent.py",
        "--candidate-solution", "candidate.py",
        "--scenario", "linear",
        "--shards", "0,1",
        "--output-dir", str(tmp_path),
    ])
    manifest = runner.run(args)
    assert load_count["count"] == 1
    assert manifest["completed_shards"] == [0, 1]
