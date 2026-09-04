"""Fast, shard-based HiF4 screening without modifying ``official_eval.py``.

``proxy-v3`` is a diagnostic overlay on the immutable proxy-v2 dense tensors.
Six balanced shards cover every Qwen layer/role exactly once.  Calibration
artifacts may be reused between in-distribution and OOD scoring, but cached
runs are explicitly ineligible for official-runtime prediction.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import pickle
import sys
import time
from pathlib import Path
from typing import Any, Mapping

import torch

try:
    import official_eval as v2
except ModuleNotFoundError as exc:  # pragma: no cover - package import path
    if exc.name != "official_eval":
        raise
    from . import official_eval as v2


PROTOCOL = "proxy-v3"
SHARD_COUNT = 6
SHARD_CASE_DESIGN = "balanced-sequential-shard-v1"
CALIBRATION_CACHE_SCHEMA = 1
CALIBRATION_CACHE_DIR = v2.CACHE_DIR / "proxy-v3-calibration"


def shard_layers(total: int, shard: int) -> tuple[int, ...]:
    if shard < 0 or shard >= SHARD_COUNT:
        raise ValueError(f"shard must be in [0, {SHARD_COUNT - 1}]")
    return tuple(range(shard, total, SHARD_COUNT))


def _in_dist_pair(raw: v2.RawPack, offset: int) -> tuple[int, ...]:
    windows = tuple(index for index in v2.COMPACT_WINDOW_INDICES if index < len(raw.test_windows))
    if len(windows) < 4:
        raise RuntimeError("proxy-v3 requires the four proxy-v2 compact holdout windows")
    start = offset % 2
    return windows[start], windows[start + 2]


def _ood_windows(raw: v2.RawPack, offset: int) -> tuple[int, ...]:
    by_domain_length: dict[tuple[str, int], list[int]] = {}
    for index, window in enumerate(raw.test_windows):
        by_domain_length.setdefault((window.split, len(window.input_ids)), []).append(index)
    domains = sorted({window.split for window in raw.test_windows})
    lengths = sorted({len(window.input_ids) for window in raw.test_windows})
    if len(domains) < 2 or not lengths:
        raise RuntimeError("proxy-v3 OOD shard requires multiple domains and lengths")
    length = lengths[offset % len(lengths)]
    selected = []
    for domain in domains:
        candidates = by_domain_length.get((domain, length), [])
        if not candidates:
            raise RuntimeError(f"OOD domain {domain!r} has no length-{length} window")
        selected.append(candidates[offset % len(candidates)])
    return tuple(selected)


def prepare_shard(raw: v2.RawPack, shard: int, scenario: str, ood: bool = False) -> v2.PreparedPack:
    if scenario not in {"both", "linear", "attention"}:
        raise ValueError(f"unsupported scenario: {scenario}")
    layers = shard_layers(raw.layers, shard)
    roles = tuple(raw.roles)
    calibration_indices = tuple(range(min(2, len(raw.calibration_windows))))
    if not calibration_indices:
        raise RuntimeError("dense pack has no Linear calibration windows")

    linear_cases: list[v2.LinearCase] = []
    if scenario in {"both", "linear"}:
        for layer_offset, layer in enumerate(layers):
            for role_offset, role in enumerate(roles):
                windows = (
                    _ood_windows(raw, shard + layer_offset + role_offset)
                    if ood else _in_dist_pair(raw, shard + layer_offset + role_offset)
                )
                for window in windows:
                    linear_cases.append(v2.LinearCase(
                        len(linear_cases), layer, role, calibration_indices, window
                    ))

    attention_cases: list[v2.AttentionCase] = []
    if scenario in {"both", "attention"}:
        for layer_offset, layer in enumerate(layers):
            windows = (
                _ood_windows(raw, shard + layer_offset)
                if ood else _in_dist_pair(raw, shard + layer_offset)
            )
            for window in windows:
                attention_cases.append(v2.AttentionCase(
                    len(attention_cases), layer,
                    tuple(range(len(raw.calibration_windows))), window,
                ))

    linear_state_keys = (
        [(layer, role) for layer in layers for role in roles]
        if scenario in {"both", "linear"} else []
    )
    attention_state_layers = list(layers) if scenario in {"both", "attention"} else []

    weights: list[dict[str, tuple[torch.Tensor, torch.Tensor]]] = [dict() for _ in range(raw.layers)]
    cal_act: dict[str, list[list[Any]]] = {
        role: [[None for _ in range(raw.layers)] for _ in raw.calibration_windows]
        for role in roles
    }
    test_act: dict[str, list[list[Any]]] = {
        role: [[None for _ in range(raw.layers)] for _ in raw.test_windows]
        for role in roles
    }
    for layer, role in linear_state_keys:
        weights[layer][role] = v2._pair(raw.weights[layer][role])
        for sample in calibration_indices:
            cal_act[role][sample][layer] = v2._pair(raw.calibration_activations[role][sample][layer])
    for case in linear_cases:
        if test_act[case.role][case.test_window][case.layer] is None:
            test_act[case.role][case.test_window][case.layer] = v2._pair(
                raw.test_activations[case.role][case.test_window][case.layer]
            )

    cal_qkv: list[list[Any]] = [
        [None for _ in range(raw.layers)] for _ in raw.calibration_windows
    ]
    test_qkv: list[list[Any]] = [
        [None for _ in range(raw.layers)] for _ in raw.test_windows
    ]
    for layer in attention_state_layers:
        for sample in range(len(raw.calibration_windows)):
            q, k, value = raw.calibration_qkv[sample][layer]
            cal_qkv[sample][layer] = {"q": v2._pair(q), "k": v2._pair(k), "v": v2._pair(value)}
    for case in attention_cases:
        if test_qkv[case.test_window][case.layer] is None:
            q, k, value = raw.test_qkv[case.test_window][case.layer]
            test_qkv[case.test_window][case.layer] = (v2._pair(q), v2._pair(k), v2._pair(value))

    metadata = dict(raw.metadata)
    metadata.update({
        "protocol": PROTOCOL,
        "base_protocol": v2.PROTOCOL,
        "case_design": SHARD_CASE_DESIGN,
        "shard": shard,
        "shard_count": SHARD_COUNT,
        "ood": ood,
        "evaluation_scenario": scenario,
        "linear_calibration_indices": list(calibration_indices),
        "linear_state_keys": [[layer, role] for layer, role in linear_state_keys],
        "attention_state_layers": attention_state_layers,
        "linear_case_count": len(linear_cases),
        "attention_case_count": len(attention_cases),
        "linear_roles": list(roles),
        "calibration_call_graph": "selected-layer-role-once; selected-attention-layer-once",
    })
    return v2.PreparedPack(
        weights, cal_act, test_act, cal_qkv, test_qkv,
        raw.calibration_windows,
        [raw.calibration_windows[index] for index in calibration_indices],
        raw.test_windows, raw.layers, raw.hidden_size, raw.q_heads, raw.kv_heads,
        raw.head_dim, linear_cases, attention_cases, metadata, roles,
    )


def _calibration_identity(path: Path, pack: v2.PreparedPack, device: torch.device) -> dict[str, Any]:
    train_path = v2.DATA_DIR / v2.WIKITEXT_FILES["train"]
    return {
        "protocol": PROTOCOL,
        "solution_sha256": v2.sha256_file(path),
        "model_revision": pack.metadata.get("model_revision"),
        "calibration_train_sha256": v2.sha256_file(train_path),
        "data_sha256": pack.metadata.get("data_sha256", {}),
        "calibration_window_keys": [
            [window.split, window.document_id, window.token_start, window.token_end]
            for window in pack.calibration_windows
        ],
        "input_codec": v2.NVFP4_INPUT_CODEC,
        "nvfp4_mode": v2.NVFP4_MODE,
        "scenario": pack.metadata["evaluation_scenario"],
        "linear_calibration_indices": pack.metadata["linear_calibration_indices"],
        "linear_state_keys": pack.metadata["linear_state_keys"],
        "attention_state_layers": pack.metadata["attention_state_layers"],
        "q_heads": pack.q_heads,
        "kv_heads": pack.kv_heads,
        "head_dim": pack.head_dim,
        "hidden_size": pack.hidden_size,
        "roles": list(pack.roles),
        "device_type": device.type,
        "device": str(device),
        # ``torch.__version__`` is a TorchVersion object on some PyTorch
        # builds.  Serialising it directly makes a weights_only artifact
        # unreadable on the next run; keep the identity JSON-safe.
        "torch_version": str(torch.__version__),
    }


def default_calibration_cache_path(identity: Mapping[str, Any]) -> Path:
    encoded = json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
    digest = hashlib.sha256(encoded).hexdigest()[:20]
    return CALIBRATION_CACHE_DIR / f"{identity['solution_sha256'][:16]}-{identity['scenario']}-{digest}.pt"


def save_calibration_artifact(
    path: Path,
    identity: Mapping[str, Any],
    weight_states: Mapping[tuple[int, str], tuple[Any, Mapping[str, torch.Tensor]]],
    attention_states: Mapping[int, Mapping[str, Any]],
) -> None:
    payload = {
        "schema": CALIBRATION_CACHE_SCHEMA,
        "identity": dict(identity),
        "weight_states": [
            {"layer": layer, "role": role, "state": state, "params": dict(params)}
            for (layer, role), (state, params) in sorted(weight_states.items())
        ],
        "attention_states": [
            {"layer": layer, "states": dict(states)}
            for layer, states in sorted(attention_states.items())
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def load_calibration_artifact(
    path: Path,
    identity: Mapping[str, Any],
    pack: v2.PreparedPack,
) -> tuple[dict[tuple[int, str], tuple[Any, dict[str, torch.Tensor]]], dict[int, dict[str, Any]]]:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, Mapping):
        raise RuntimeError("calibration artifact payload is not a mapping")
    if payload.get("schema") != CALIBRATION_CACHE_SCHEMA:
        raise RuntimeError("calibration artifact schema mismatch")
    if dict(payload.get("identity", {})) != dict(identity):
        raise RuntimeError("calibration artifact identity mismatch")

    weight_states: dict[tuple[int, str], tuple[Any, dict[str, torch.Tensor]]] = {}
    for item in payload.get("weight_states", []):
        key = int(item["layer"]), str(item["role"])
        if key in weight_states:
            raise RuntimeError(f"calibration artifact contains duplicate weight state {key}")
        state = item["state"]
        params = dict(item["params"])
        v2.validate_state(state)
        reference_shape = v2.dequantize_nvfp4(*pack.weights[key[0]][key[1]]).shape
        v2.validate_hif4_params(params, reference_shape)
        weight_states[key] = state, params
    attention_states: dict[int, dict[str, Any]] = {}
    for item in payload.get("attention_states", []):
        layer = int(item["layer"])
        if layer in attention_states:
            raise RuntimeError(f"calibration artifact contains duplicate attention state {layer}")
        states = dict(item["states"])
        if set(states) != {"q_state", "k_state", "v_state"}:
            raise RuntimeError("cached attention state has unexpected keys")
        for name in states:
            v2.validate_state(states[name])
        attention_states[layer] = states

    expected_weights = {
        (int(layer), str(role)) for layer, role in pack.metadata["linear_state_keys"]
    }
    expected_attention = {int(layer) for layer in pack.metadata["attention_state_layers"]}
    if set(weight_states) != expected_weights or set(attention_states) != expected_attention:
        raise RuntimeError("calibration artifact state coverage mismatch")
    return weight_states, attention_states


def _calibrate(
    solution: Any,
    pack: v2.PreparedPack,
    device: torch.device,
) -> tuple[
    dict[tuple[int, str], tuple[Any, dict[str, torch.Tensor]]],
    dict[int, dict[str, Any]],
    dict[str, float],
    dict[str, int],
]:
    seconds = {name: 0.0 for name in v2.REQUIRED_APIS}
    calls = {name: 0 for name in v2.REQUIRED_APIS}
    weight_states: dict[tuple[int, str], tuple[Any, dict[str, torch.Tensor]]] = {}
    for layer, role in ((int(item[0]), str(item[1])) for item in pack.metadata["linear_state_keys"]):
        weight_pair = v2._move_pair(pack.weights[layer][role], device)
        calibration = [
            v2._move_pair(pack.linear_calibration_activations[role][sample][layer], device)
            for sample in pack.metadata["linear_calibration_indices"]
        ]
        started = time.perf_counter()
        result = solution.hif4_calibration_and_quantize_weight(weight_pair[0], weight_pair[1], calibration)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        seconds["hif4_calibration_and_quantize_weight"] += time.perf_counter() - started
        calls["hif4_calibration_and_quantize_weight"] += 1
        if not isinstance(result, Mapping) or set(result) != {"weight_params", "activation_state"}:
            raise ValueError("weight calibration returned an invalid mapping")
        v2.validate_state(result["activation_state"])
        shape = v2.dequantize_nvfp4(*pack.weights[layer][role]).shape
        v2.validate_hif4_params(result["weight_params"], shape)
        weight_states[(layer, role)] = result["activation_state"], v2._cpu_params(result["weight_params"])

    attention_states: dict[int, dict[str, Any]] = {}
    for layer in (int(value) for value in pack.metadata["attention_state_layers"]):
        calibration = [
            v2._move_qkv(pack.calibration_qkv[sample][layer], device)
            for sample in range(len(pack.calibration_windows))
        ]
        started = time.perf_counter()
        states = solution.hif4_calibration_attention(
            calibration, pack.q_heads, pack.kv_heads, pack.head_dim
        )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        seconds["hif4_calibration_attention"] += time.perf_counter() - started
        calls["hif4_calibration_attention"] += 1
        if not isinstance(states, Mapping) or set(states) != {"q_state", "k_state", "v_state"}:
            raise ValueError("attention calibration returned an invalid mapping")
        for name in states:
            v2.validate_state(states[name])
        attention_states[layer] = dict(states)
    return weight_states, attention_states, seconds, calls


def _score(
    solution: Any,
    pack: v2.PreparedPack,
    device: torch.device,
    weight_states: Mapping[tuple[int, str], tuple[Any, Mapping[str, torch.Tensor]]],
    attention_states: Mapping[int, Mapping[str, Any]],
    seconds: dict[str, float],
    calls: dict[str, int],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    linear_details: list[dict[str, Any]] = []
    standard_weight_cache: dict[tuple[int, str], tuple[torch.Tensor, torch.Tensor]] = {}
    for case in pack.linear_cases:
        state, weight_params = weight_states[(case.layer, case.role)]
        activation_pair = v2._move_pair(pack.test_activations[case.role][case.test_window][case.layer], device)
        started = time.perf_counter()
        activation_params = solution.hif4_dynamic_quantize_activation(
            activation_pair[0], activation_pair[1], state
        )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        seconds["hif4_dynamic_quantize_activation"] += time.perf_counter() - started
        calls["hif4_dynamic_quantize_activation"] += 1
        ref_activation = v2.dequantize_nvfp4(
            *pack.test_activations[case.role][case.test_window][case.layer]
        ).to(torch.float32)
        key = case.layer, case.role
        if key not in standard_weight_cache:
            ref_weight = v2.dequantize_nvfp4(*pack.weights[case.layer][case.role]).to(torch.float32)
            standard_weight_cache[key] = (
                ref_weight,
                v2.decode_standard_hif4(v2.encode_standard_hif4(ref_weight)).to(torch.float32),
            )
        ref_weight, standard_weight = standard_weight_cache[key]
        standard_activation = v2.decode_standard_hif4(
            v2.encode_standard_hif4(ref_activation)
        ).to(torch.float32)
        v2.validate_hif4_params(activation_params, ref_activation.shape)
        player_activation = v2.dequantize_hif4(v2._cpu_params(activation_params), ref_activation.shape).to(torch.float32)
        player_weight = v2.dequantize_hif4(dict(weight_params), ref_weight.shape).to(torch.float32)
        reference = ref_activation.to(device) @ ref_weight.to(device).T
        standard = standard_activation.to(device) @ standard_weight.to(device).T
        player = player_activation.to(device) @ player_weight.to(device).T
        details = v2._score_details(standard, player, reference)
        details.update({
            "case_id": case.case_id,
            "layer": case.layer,
            "role": case.role,
            "role_family": v2._linear_role_family(case.role),
            "calibration_indices": list(case.calibration_indices),
            "test_window": case.test_window,
            "test_split": pack.test_windows[case.test_window].split,
            "test_length": len(pack.test_windows[case.test_window].input_ids),
            "input_width": int(ref_activation.shape[-1]),
            "output_width": int(ref_weight.shape[0]),
            "shape_bucket": (
                "hidden_to_hidden"
                if ref_activation.shape[-1] == pack.hidden_size and ref_weight.shape[0] == pack.hidden_size
                else "hidden_to_wide" if ref_activation.shape[-1] == pack.hidden_size
                else "wide_to_hidden" if ref_weight.shape[0] == pack.hidden_size else "other"
            ),
        })
        linear_details.append(details)

    attention_details: list[dict[str, Any]] = []
    for case in pack.attention_cases:
        states = attention_states[case.layer]
        pairs = pack.test_qkv[case.test_window][case.layer]
        q_pair, k_pair, value_pair = (v2._move_pair(pair, device) for pair in pairs)
        outputs = []
        for api_name, function, pair, heads, state_name in (
            ("hif4_dynamic_quantize_q", solution.hif4_dynamic_quantize_q, q_pair, pack.q_heads, "q_state"),
            ("hif4_dynamic_quantize_k", solution.hif4_dynamic_quantize_k, k_pair, pack.kv_heads, "k_state"),
            ("hif4_dynamic_quantize_v", solution.hif4_dynamic_quantize_v, value_pair, pack.kv_heads, "v_state"),
        ):
            started = time.perf_counter()
            params = function(pair[0], pair[1], heads, pack.head_dim, states[state_name])
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            seconds[api_name] += time.perf_counter() - started
            calls[api_name] += 1
            outputs.append(params)
        references = [v2.dequantize_nvfp4(*pair).to(torch.float32) for pair in pairs]
        standards = [
            v2.decode_standard_hif4(v2.encode_standard_hif4(value)).to(torch.float32)
            for value in references
        ]
        players = []
        for params, reference in zip(outputs, references):
            v2.validate_hif4_params(params, reference.shape)
            players.append(v2.dequantize_hif4(v2._cpu_params(params), reference.shape).to(torch.float32))
        reference_output = v2._attention(
            *(value.to(device)[None] for value in references),
            pack.q_heads, pack.kv_heads, pack.head_dim,
        )
        standard_output = v2._attention(
            *(value.to(device)[None] for value in standards),
            pack.q_heads, pack.kv_heads, pack.head_dim,
        )
        player_output = v2._attention(
            *(value.to(device)[None] for value in players),
            pack.q_heads, pack.kv_heads, pack.head_dim,
        )
        details = v2._score_details(standard_output, player_output, reference_output)
        details.update({
            "case_id": case.case_id,
            "layer": case.layer,
            "calibration_indices": list(case.calibration_indices),
            "test_window": case.test_window,
            "test_split": pack.test_windows[case.test_window].split,
            "test_length": int(references[0].shape[0]),
        })
        attention_details.append(details)
    return linear_details, attention_details


def evaluate(
    path: Path,
    pack: v2.PreparedPack,
    device_name: str,
    calibration_cache_mode: str = "auto",
    calibration_cache: Path | None = None,
) -> dict[str, Any]:
    solution = v2.load_solution(path)
    device = torch.device(device_name)
    identity = _calibration_identity(path, pack, device)
    cache_path = calibration_cache or default_calibration_cache_path(identity)
    cache_hit = False
    calibration_wall_start = time.perf_counter()
    if calibration_cache_mode in {"auto", "read"} and cache_path.is_file():
        try:
            weight_states, attention_states = load_calibration_artifact(cache_path, identity, pack)
        except (OSError, RuntimeError, ValueError, KeyError, TypeError, pickle.UnpicklingError):
            if calibration_cache_mode == "read":
                raise
            # ``auto`` treats an unreadable/old artifact exactly like a stale
            # identity: recalibrate and atomically replace it below.  This is
            # important when PyTorch changes its weights_only allow-list.
            weight_states, attention_states, seconds, calls = _calibrate(solution, pack, device)
            if calibration_cache_mode in {"auto", "write"}:
                save_calibration_artifact(cache_path, identity, weight_states, attention_states)
        else:
            seconds = {name: 0.0 for name in v2.REQUIRED_APIS}
            calls = {name: 0 for name in v2.REQUIRED_APIS}
            cache_hit = True
    elif calibration_cache_mode == "read":
        raise FileNotFoundError(f"calibration artifact does not exist: {cache_path}")
    else:
        weight_states, attention_states, seconds, calls = _calibrate(solution, pack, device)
        if calibration_cache_mode in {"auto", "write"}:
            save_calibration_artifact(cache_path, identity, weight_states, attention_states)
    calibration_elapsed = time.perf_counter() - calibration_wall_start
    calibration_wall = calibration_elapsed if not cache_hit else 0.0
    cache_load_wall = calibration_elapsed if cache_hit else 0.0

    wall_start = time.perf_counter()
    linear_details, attention_details = _score(
        solution, pack, device, weight_states, attention_states, seconds, calls
    )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    scoring_wall = time.perf_counter() - wall_start
    linear_gains = [float(item["gain"]) for item in linear_details]
    attention_gains = [float(item["gain"]) for item in attention_details]
    total_cases = len(linear_gains) + len(attention_gains)
    calibration_api_names = {
        "hif4_calibration_and_quantize_weight", "hif4_calibration_attention"
    }
    scoring_api_seconds = sum(
        seconds[name] for name in v2.REQUIRED_APIS if name not in calibration_api_names
    )
    calibration_api_seconds = sum(seconds[name] for name in calibration_api_names)
    result = {
        "candidate": path.stem,
        "status": "ok",
        "source": str(path.resolve()),
        "source_sha256": identity["solution_sha256"],
        "evaluation_scope": {
            "kind": f"{pack.metadata['evaluation_scenario']}-only-sequential-shard"
            if pack.metadata["evaluation_scenario"] != "both" else "sequential-shard",
            "intent": "fast-paired-screening-and-failure-localization",
            "comparable_for_proxy_ranking": False,
            "paired_only": True,
            "official_score_equivalent": False,
            "shard": pack.metadata["shard"],
            "shard_count": SHARD_COUNT,
        },
        "score": {
            "linear_mean": sum(linear_gains) / len(linear_gains) if linear_gains else 0.0,
            "attention_mean": sum(attention_gains) / len(attention_gains) if attention_gains else 0.0,
            "overall_mean": (sum(linear_gains) + sum(attention_gains)) / total_cases if total_cases else 0.0,
            "linear_cases": len(linear_gains),
            "attention_cases": len(attention_gains),
        },
        "timing": {
            "api_seconds": seconds,
            "api_total_seconds": float(sum(seconds.values())),
            "api_calls": calls,
            "scoring_wall_seconds": scoring_wall,
            "calibration_wall_seconds": calibration_wall,
            "calibration_cache_load_seconds": cache_load_wall,
            "calibration_api_seconds": calibration_api_seconds,
            "scoring_api_seconds": scoring_api_seconds,
            "calibration_cache_hit": cache_hit,
            "calibration_artifact": str(cache_path.resolve()),
            "calibration_identity": identity,
            "calibration_timing_measured": not cache_hit,
            "official_time_predictable": False,
            "note": "shard/cache timings are diagnostic only; use fresh default official_eval for time audit",
        },
        "case_scores": {"linear": linear_details, "attention": attention_details},
        "diagnostic_config": {
            "evaluation_scenario": pack.metadata["evaluation_scenario"],
            "error_source_decomposition": False,
            "shard": pack.metadata["shard"],
            "ood": pack.metadata["ood"],
        },
    }
    return result


def _write_report(path: Path, output: Mapping[str, Any]) -> None:
    result = output["results"][0]
    score = result["score"]
    timing = result["timing"]
    paired = output.get("paired_effect")
    lines = [
        "# proxy-v3 shard result", "",
        f"- shard: `{output['shard']}/{SHARD_COUNT - 1}`; scenario: `{output['scenario']}`; OOD: `{output['ood']}`",
        f"- Linear/Attention cases: `{score['linear_cases']}/{score['attention_cases']}`",
        f"- Linear/Attention mean: `{score['linear_mean']:.6f}/{score['attention_mean']:.6f}`",
        f"- API seconds: `{timing['api_total_seconds']:.3f}`; calibration cache hit: `{timing['calibration_cache_hit']}`",
        f"- calibration wall/API: `{timing['calibration_wall_seconds']:.3f}/{timing['calibration_api_seconds']:.3f}s`; "
        f"cache load: `{timing['calibration_cache_load_seconds']:.3f}s`; scoring API: "
        f"`{timing['scoring_api_seconds']:.3f}s`; scoring wall: `{timing['scoring_wall_seconds']:.3f}s`",
        "- official score/time equivalent: `false`",
    ]
    if paired and paired.get("enabled"):
        lines.extend(["", "## Paired delta", ""])
        for side in ("linear", "attention"):
            item = paired[side]["overall"]
            if item["case_count"]:
                lines.append(
                    f"- {side}: mean `{item['mean_delta_gain']:+.6f}`, median "
                    f"`{item['median_delta_gain']:+.6f}`, +/-/0 "
                    f"`{item['positive_cases']}/{item['negative_cases']}/{item['zero_cases']}`"
                )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _load_eval_document(path: Path) -> dict[str, Any]:
    """Read a v2 or v3 result for paired diagnostics without changing v2."""
    source = path.resolve()
    if not source.is_file():
        raise FileNotFoundError(f"evaluation JSON does not exist: {source}")
    document = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(document, dict) or not isinstance(document.get("results"), list):
        raise ValueError(f"evaluation JSON has no results list: {source}")
    if document.get("protocol") not in {v2.PROTOCOL, PROTOCOL}:
        raise ValueError(f"unsupported evaluation protocol {document.get('protocol')!r}: {source}")
    return document


def make_output(
    cache_path: Path,
    pack: v2.PreparedPack,
    prepare_seconds: float,
    result: Mapping[str, Any],
    paired: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Wrap one evaluated shard in the on-disk proxy-v3 document schema."""
    return {
        "protocol": PROTOCOL,
        "base_protocol": v2.PROTOCOL,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "data_source": "proxy-v2-dense-cache",
        "dense_cache": str(cache_path.resolve()),
        "prepare_seconds": prepare_seconds,
        "shard": pack.metadata["shard"],
        "scenario": pack.metadata["evaluation_scenario"],
        "ood": pack.metadata["ood"],
        "data_metadata": pack.metadata,
        "results": [dict(result)],
        "paired_effect": paired,
    }


def write_output(path: Path, output: Mapping[str, Any], report: Path | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(output, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    if report:
        _write_report(report, output)


def cleanup_solution_modules() -> None:
    for module_name in list(sys.modules):
        if module_name.startswith("_hif4_official_"):
            sys.modules.pop(module_name, None)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def run(args: argparse.Namespace) -> dict[str, Any]:
    cache_path = (args.cache or (v2.OOD_CACHE if args.ood else v2.DEFAULT_CACHE)).resolve()
    if not cache_path.is_file():
        raise FileNotFoundError(
            f"dense proxy-v2 cache does not exist: {cache_path}; build it with official_eval.py first"
        )
    started = time.perf_counter()
    raw = v2.load_pack(cache_path)
    pack = prepare_shard(raw, args.shard, args.evaluation_scenario, args.ood)
    # ``load_pack`` materializes the full dense snapshot.  The prepared shard
    # owns only the selected encoded pairs, so release the multi-GB raw banks
    # before importing and running a candidate.
    del raw
    gc.collect()
    prepare_seconds = time.perf_counter() - started
    result = evaluate(
        args.solution.resolve(), pack, args.algorithm_device,
        args.calibration_cache_mode,
        args.calibration_cache.resolve() if args.calibration_cache else None,
    )
    paired = None
    if args.baseline_json:
        baseline_doc = _load_eval_document(args.baseline_json)
        baseline = v2._select_eval_result(baseline_doc, args.baseline_result, "baseline")
        paired = v2._paired_effect_diagnostics(
            baseline, result, v2._focus_selectors(args.focus_linear_roles)
        )
    output = make_output(cache_path, pack, prepare_seconds, result, paired)
    write_output(args.output, output, args.report)
    cleanup_solution_modules()
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--solution", type=Path, required=True)
    parser.add_argument("--shard", type=int, choices=range(SHARD_COUNT), required=True)
    scenario = parser.add_mutually_exclusive_group()
    scenario.add_argument("--linear-only", dest="evaluation_scenario", action="store_const", const="linear")
    scenario.add_argument("--attention-only", dest="evaluation_scenario", action="store_const", const="attention")
    parser.set_defaults(evaluation_scenario="both")
    parser.add_argument("--ood", action="store_true")
    parser.add_argument("--cache", type=Path)
    parser.add_argument("--algorithm-device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--calibration-cache-mode", choices=("off", "auto", "read", "write"), default="auto"
    )
    parser.add_argument("--calibration-cache", type=Path)
    parser.add_argument("--baseline-json", type=Path)
    parser.add_argument("--baseline-result")
    parser.add_argument("--focus-linear-roles", default="")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
