from __future__ import annotations

import time
from typing import Iterable, Sequence

import torch

from .compliance import validate_state
from .formats import (
    dequantize_hif4,
    dequantize_nvfp4,
    standard_hif4_quantize,
    validate_hif4_params,
)
from .models import CaseResult, RunResult, TimingResult
from .scoring import attention_output, competition_score
from .solution_loader import SolutionAPI
from .suites import AttentionCase, EvaluationSuite, LinearCase, Pair


def _move_pair(pair: Pair, device: torch.device) -> Pair:
    return pair[0].to(device), pair[1].to(device)


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _mse(reference: torch.Tensor, value: torch.Tensor) -> float:
    return float(torch.mean((value.to(torch.float32) - reference.to(torch.float32)).square()))


def _linear_cases(
    api: SolutionAPI,
    cases: Iterable[LinearCase],
    device: torch.device,
    dtype_name: str,
) -> tuple[list[CaseResult], float, float]:
    results: list[CaseResult] = []
    player_seconds = 0.0
    wall_start = time.perf_counter()
    for case in cases:
        weight_pair = _move_pair(case.weight, device)
        weight_dense = dequantize_nvfp4(*weight_pair)
        calibration = [_move_pair(pair, device) for pair in case.calibration]
        _sync(device)
        start = time.perf_counter()
        calibrated = api.function("hif4_calibration_and_quantize_weight")(
            weight_pair[0], weight_pair[1], calibration
        )
        _sync(device)
        player_seconds += time.perf_counter() - start
        if not isinstance(calibrated, dict) or set(calibrated) != {"weight_params", "activation_state"}:
            raise ValueError("weight calibration must return weight_params and activation_state")
        validate_state(calibrated["activation_state"])
        candidate_weight = calibrated["weight_params"]
        validate_hif4_params(candidate_weight, weight_dense.shape)
        candidate_weight_dense = dequantize_hif4(candidate_weight)
        standard_weight = dequantize_hif4(standard_hif4_quantize(weight_dense))
        for test_index, raw_pair in enumerate(case.tests):
            test_pair = _move_pair(raw_pair, device)
            activation_dense = dequantize_nvfp4(*test_pair)
            standard_activation = dequantize_hif4(standard_hif4_quantize(activation_dense))
            _sync(device)
            start = time.perf_counter()
            candidate_activation = api.function("hif4_dynamic_quantize_activation")(
                test_pair[0], test_pair[1], calibrated["activation_state"]
            )
            _sync(device)
            player_seconds += time.perf_counter() - start
            validate_hif4_params(candidate_activation, activation_dense.shape)
            candidate_activation_dense = dequantize_hif4(candidate_activation)
            reference = activation_dense.to(torch.float32) @ weight_dense.to(torch.float32).transpose(-1, -2)
            standard_output = standard_activation.to(torch.float32) @ standard_weight.to(torch.float32).transpose(-1, -2)
            player_output = candidate_activation_dense.to(torch.float32) @ candidate_weight_dense.to(torch.float32).transpose(-1, -2)
            results.append(
                CaseResult(
                    "seed-00", "linear", case.name, test_index, False, dtype_name,
                    _mse(reference, standard_output), _mse(reference, player_output),
                    competition_score(reference, standard_output, player_output),
                )
            )
    return results, player_seconds, time.perf_counter() - wall_start


def _attention_cases(
    api: SolutionAPI,
    cases: Iterable[AttentionCase],
    device: torch.device,
    dtype_name: str,
    causal_modes: Sequence[bool],
) -> tuple[list[CaseResult], float, float]:
    results: list[CaseResult] = []
    player_seconds = 0.0
    wall_start = time.perf_counter()
    for case in cases:
        calibration_list = [
            {key: _move_pair(value, device) for key, value in sample.items()}
            for sample in case.calibration
        ]
        _sync(device)
        start = time.perf_counter()
        states = api.function("hif4_calibration_attention")(
            calibration_list, case.q_num_heads, case.kv_num_heads, case.head_dim
        )
        _sync(device)
        player_seconds += time.perf_counter() - start
        if not isinstance(states, dict) or set(states) != {"q_state", "k_state", "v_state"}:
            raise ValueError("attention calibration must return q_state, k_state, and v_state")
        validate_state(states["q_state"])
        validate_state(states["k_state"])
        validate_state(states["v_state"])
        for test_index, raw_sample in enumerate(case.tests):
            sample = {key: _move_pair(value, device) for key, value in raw_sample.items()}
            dense = {key: dequantize_nvfp4(*value) for key, value in sample.items()}
            standard = {
                key: dequantize_hif4(standard_hif4_quantize(value))
                for key, value in dense.items()
            }
            _sync(device)
            start = time.perf_counter()
            q_params = api.function("hif4_dynamic_quantize_q")(
                sample["q"][0], sample["q"][1], case.q_num_heads, case.head_dim, states["q_state"]
            )
            k_params = api.function("hif4_dynamic_quantize_k")(
                sample["k"][0], sample["k"][1], case.kv_num_heads, case.head_dim, states["k_state"]
            )
            v_params = api.function("hif4_dynamic_quantize_v")(
                sample["v"][0], sample["v"][1], case.kv_num_heads, case.head_dim, states["v_state"]
            )
            _sync(device)
            player_seconds += time.perf_counter() - start
            validate_hif4_params(q_params, dense["q"].shape)
            validate_hif4_params(k_params, dense["k"].shape)
            validate_hif4_params(v_params, dense["v"].shape)
            player = {
                key: dequantize_hif4(value)
                for key, value in {"q": q_params, "k": k_params, "v": v_params}.items()
            }
            for causal in causal_modes:
                reference = attention_output(
                    dense["q"], dense["k"], dense["v"],
                    case.q_num_heads, case.kv_num_heads, case.head_dim, causal,
                )
                standard_output = attention_output(
                    standard["q"], standard["k"], standard["v"],
                    case.q_num_heads, case.kv_num_heads, case.head_dim, causal,
                )
                player_output = attention_output(
                    player["q"], player["k"], player["v"],
                    case.q_num_heads, case.kv_num_heads, case.head_dim, causal,
                )
                results.append(
                    CaseResult(
                        "seed-00", "attention", case.name, test_index, causal, dtype_name,
                        _mse(reference, standard_output), _mse(reference, player_output),
                        competition_score(reference, standard_output, player_output),
                    )
                )
    return results, player_seconds, time.perf_counter() - wall_start


def evaluate_solution(
    api: SolutionAPI,
    suite: EvaluationSuite,
    device: torch.device,
    compute_dtypes: Sequence[str],
    causal_modes: Sequence[bool],
) -> RunResult:
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")
    all_cases: list[CaseResult] = []
    player_seconds = 0.0
    wall_seconds = 0.0
    with torch.no_grad():
        for dtype_name in compute_dtypes:
            linear, linear_player, linear_wall = _linear_cases(api, suite.linear, device, dtype_name)
            attention, attention_player, attention_wall = _attention_cases(api, suite.attention, device, dtype_name, causal_modes)
            all_cases.extend(linear)
            all_cases.extend(attention)
            player_seconds += linear_player + attention_player
            wall_seconds += linear_wall + attention_wall
    return RunResult(
        tuple(all_cases),
        TimingResult(player_seconds, wall_seconds),
        {"device": str(device), "case_count": len(all_cases), "candidate_sha256": api.sha256},
    )


