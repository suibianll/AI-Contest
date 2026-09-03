"""Audit the fixed C cross-fold Linear candidate on cached compact states."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import torch


ROOT = Path(__file__).resolve().parents[1]
EVALUATOR = ROOT / "evaluator"
sys.path.insert(0, str(EVALUATOR))

import official_eval  # noqa: E402
from reference_hif4 import validate_hif4_params, validate_state  # noqa: E402


PARENT = (
    ROOT
    / "solutions/20260903_v160_v159-linear-l1batch_v158-attn-a2_scoreNA_timeNA/solution.py"
)
CANDIDATE = ROOT / "workbench/score21765_linear_crossfold_minimax.py"
EXPECTED_PARENT_SHA = "33B1D061CE6BFCD92659C597BE4830BB9B910E646FF518433DA67B925AE8680D"
FOLD_COUNT = 5
MAX_ROWS = 128


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _equal(left: Any, right: Any) -> bool:
    if torch.is_tensor(left) or torch.is_tensor(right):
        return (
            torch.is_tensor(left)
            and torch.is_tensor(right)
            and left.dtype == right.dtype
            and tuple(left.shape) == tuple(right.shape)
            and torch.equal(left, right)
        )
    if isinstance(left, dict) or isinstance(right, dict):
        return (
            isinstance(left, dict)
            and isinstance(right, dict)
            and set(left) == set(right)
            and all(_equal(left[key], right[key]) for key in left)
        )
    if isinstance(left, (tuple, list)) or isinstance(right, (tuple, list)):
        return (
            type(left) is type(right)
            and len(left) == len(right)
            and all(_equal(a, b) for a, b in zip(left, right))
        )
    return left == right


def _transformed_exact(module, pair, state, max_rows: int) -> torch.Tensor:
    quant = module._sample_rows(pair[0], max_rows)
    scale = module._sample_rows(pair[1], max_rows)
    dense = module._dequantize_nvfp4_float32(quant, scale)
    smooth_inv = state.get("smooth_inv")
    if smooth_inv is not None:
        dense = dense * smooth_inv.to(dense.device).reshape(1, -1)
    permutation = state.get("permutation")
    if permutation is not None:
        dense = dense.index_select(-1, permutation.to(dense.device))
    block_size = int(state.get("block_smooth_size", 0))
    if block_size:
        dense = module._block_hadamard_transform(
            dense, block_size, int(state.get("block_smooth_seed", 0))
        )
    return dense.to(torch.float32)


def _fold_metrics(module, weight_pair, calibration_pairs, result) -> dict[str, Any]:
    state = result["activation_state"]
    parent_weight = module._dequantize_hif4(result["weight_params"]).to(torch.float32)
    dense_weight = module._dequantize_nvfp4_float32(*weight_pair)
    channels = int(dense_weight.shape[1])
    smooth_inv = state.get("smooth_inv")
    d = (
        torch.ones(channels, dtype=torch.float32, device=dense_weight.device)
        if smooth_inv is None
        else smooth_inv.to(dense_weight.device).reciprocal()
    )
    permutation = state.get("permutation")
    order = (
        torch.arange(channels, device=dense_weight.device)
        if permutation is None
        else permutation.to(dense_weight.device)
    )
    transformed_weight = module._linear_pair_transform(
        dense_weight,
        d,
        order,
        int(state.get("block_smooth_size", 0)),
        int(state.get("block_smooth_seed", 0)),
        weight_side=True,
    )
    exact_windows = []
    quantized_windows = []
    for pair in calibration_pairs:
        exact_windows.append(
            _transformed_exact(module, pair, state, MAX_ROWS)
        )
        quant = module._sample_rows(pair[0], MAX_ROWS)
        scale = module._sample_rows(pair[1], MAX_ROWS)
        params = module.hif4_dynamic_quantize_activation(quant, scale, state)
        quantized_windows.append(module._dequantize_hif4(params).to(torch.float32))

    records = []
    for fold in range(FOLD_COUNT):
        exact = torch.cat(
            [window[fold::FOLD_COUNT] for window in exact_windows]
        )
        quantized = torch.cat(
            [window[fold::FOLD_COUNT] for window in quantized_windows]
        )
        target = exact.mm(transformed_weight.t())
        a_only = quantized.mm(transformed_weight.t())
        w_only = exact.mm(parent_weight.t())
        both = quantized.mm(parent_weight.t())
        energy = target.square().sum(dim=0).clamp_min(module._EPS)

        def ratio(value: torch.Tensor) -> float:
            return float(((value - target).square().sum(dim=0) / energy).mean().item())

        a_ratio = ratio(a_only)
        w_ratio = ratio(w_only)
        both_ratio = ratio(both)
        records.append(
            {
                "fold": fold,
                "rows": int(exact.shape[0]),
                "a_only_ratio": a_ratio,
                "w_only_ratio": w_ratio,
                "both_ratio": both_ratio,
                "interaction_ratio": both_ratio - a_ratio - w_ratio,
            }
        )
    both_values = [record["both_ratio"] for record in records]
    return {
        "folds": records,
        "both_mean": sum(both_values) / len(both_values),
        "both_median": statistics.median(both_values),
        "both_worst": max(both_values),
        "both_variance": statistics.pvariance(both_values),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-states", type=int, default=1)
    parser.add_argument("--baseline-json", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "artifacts/official_eval/score21765-c01-linear-audit.json",
    )
    args = parser.parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the registered C audit")
    parent_hash = hashlib.sha256(PARENT.read_bytes()).hexdigest().upper()
    if parent_hash != EXPECTED_PARENT_SHA:
        raise ValueError(f"unexpected parent SHA: {parent_hash}")

    profile = official_eval._nvfp4_cache_profile(
        linear_count=None,
        attention_count=None,
        full_cases=False,
        effect_panel=False,
        compact_panel=True,
        evaluation_scenario="linear",
    )
    source_cache = official_eval.DEFAULT_CACHE
    nvfp4_cache = official_eval._default_nvfp4_cache_path(source_cache, profile)
    pack = official_eval.load_nvfp4_cache(nvfp4_cache, source_cache, profile)
    parent = _load(PARENT, "score21765_c_parent")
    candidate = _load(CANDIDATE, "score21765_c_candidate")
    candidate._WEIGHT_CROSSFOLD_DIAGNOSTICS.clear()
    calibration_indices = [int(value) for value in pack.metadata["linear_calibration_indices"]]
    state_keys = [
        (int(layer), str(role)) for layer, role in pack.metadata["linear_state_keys"]
    ][: args.max_states]
    records = []
    baseline_records = {}
    if args.baseline_json is not None:
        baseline = json.loads(args.baseline_json.read_text(encoding="utf-8"))
        baseline_records = {
            (int(record["layer"]), str(record["role"])): record
            for record in baseline["records"]
        }

    for layer, role in state_keys:
        weight_pair = official_eval._move_pair(pack.weights[layer][role], device)
        calibration_pairs = [
            official_eval._move_pair(
                pack.linear_calibration_activations[role][index][layer], device
            )
            for index in calibration_indices
        ]
        parent_result = None
        baseline_record = baseline_records.get((layer, role))
        if baseline_record is None:
            started = time.perf_counter()
            parent_result = parent.hif4_calibration_and_quantize_weight(
                weight_pair[0], weight_pair[1], calibration_pairs
            )
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            parent_seconds = time.perf_counter() - started
            parent_metrics = _fold_metrics(
                parent, weight_pair, calibration_pairs, parent_result
            )
        else:
            parent_seconds = float(baseline_record["parent_seconds"])
            parent_metrics = baseline_record["parent"]

        started = time.perf_counter()
        candidate_result = candidate.hif4_calibration_and_quantize_weight(
            weight_pair[0], weight_pair[1], calibration_pairs
        )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        candidate_seconds = time.perf_counter() - started
        validate_state(candidate_result["activation_state"])
        validate_hif4_params(
            candidate_result["weight_params"],
            official_eval.dequantize_nvfp4(*weight_pair).shape,
        )
        candidate_metrics = _fold_metrics(
            candidate, weight_pair, calibration_pairs, candidate_result
        )
        diagnostic = dict(candidate._WEIGHT_CROSSFOLD_DIAGNOSTICS[-1])
        if parent_result is None:
            hierarchy_equal = None
            actual_changed_codes = int(diagnostic["changed_codes"])
            diagnostic_changed_codes_equal = None
            activation_state_equal = None
        else:
            hierarchy_equal = all(
                torch.equal(
                    parent_result["weight_params"][name],
                    candidate_result["weight_params"][name],
                )
                for name in ("scale_factor", "scale_lv2", "scale_lv3")
            )
            parent_codes = (
                parent_result["weight_params"]["sign"]
                * parent_result["weight_params"]["mant"]
                * 4.0
            )
            candidate_codes = (
                candidate_result["weight_params"]["sign"]
                * candidate_result["weight_params"]["mant"]
                * 4.0
            )
            actual_changed_codes = int((parent_codes != candidate_codes).sum().item())
            diagnostic_changed_codes_equal = actual_changed_codes == int(
                diagnostic["changed_codes"]
            )
            activation_state_equal = _equal(
                parent_result["activation_state"], candidate_result["activation_state"]
            )
        record = {
            "layer": layer,
            "role": role,
            "weight_shape": list(official_eval.dequantize_nvfp4(*weight_pair).shape),
            "parent_seconds": parent_seconds,
            "candidate_seconds": candidate_seconds,
            "calibration_ratio": candidate_seconds / max(parent_seconds, 1.0e-12),
            "activation_state_equal": activation_state_equal,
            "hierarchy_equal": hierarchy_equal,
            "actual_changed_codes": actual_changed_codes,
            "diagnostic_changed_codes_equal": diagnostic_changed_codes_equal,
            "parent": parent_metrics,
            "candidate": candidate_metrics,
            "diagnostic": diagnostic,
        }
        records.append(record)
        print(
            f"layer={layer} role={role} parent={parent_seconds:.3f}s "
            f"candidate={candidate_seconds:.3f}s ratio={record['calibration_ratio']:.3f} "
            f"accepted={diagnostic['accepted_blocks']}/{diagnostic['attempted_blocks']} "
            f"changed={actual_changed_codes}",
            flush=True,
        )

    output = {
        "experiment": "score21765-c01-crossfold-minimax",
        "scope": "linear-compact-selected-calibration-states-audit",
        "parent": str(PARENT.relative_to(ROOT)),
        "parent_sha256": parent_hash,
        "candidate": str(CANDIDATE.relative_to(ROOT)),
        "candidate_sha256": hashlib.sha256(CANDIDATE.read_bytes()).hexdigest().upper(),
        "fold_definition": "sample each window to <=128 rows, then merge row_index mod 5 across both windows",
        "calibration_indices": calibration_indices,
        "device": str(device),
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
