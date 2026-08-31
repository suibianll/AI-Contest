"""Calibration-only diagnostics for the L3 Global Activation-LRH gate.

The script intentionally never reads test activations.  It reconstructs the
same transformed calibration tensors used by the Linear API, asks the
candidate generator for its exact deployed-Gram decision, and records only
proposal/acceptance statistics for each calibration fold.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
import time
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from nvfp4_sim import nvfp4_encode  # noqa: E402


DEFAULT_LAYERS = (0, 5, 11, 17, 23)
DEFAULT_ROLES = ("q", "k", "v", "o", "fc_gate", "fc_up", "proj")


def _load_solution(path: Path):
    spec = importlib.util.spec_from_file_location("l3_activation_lrh_solution", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load solution from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _sha256_lf(path: Path) -> str:
    data = path.read_bytes().replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    return hashlib.sha256(data).hexdigest()


def _fold_diagnostics(solution, dense: torch.Tensor, state: dict[str, Any]) -> dict[str, Any]:
    smooth_inv = state.get("smooth_inv")
    if torch.is_tensor(smooth_inv):
        dense = dense * smooth_inv.to(dense.device).reshape(1, -1)
    seed = int(state.get("block_smooth_seed", -1))
    block_size = int(state.get("block_smooth_size", 0))
    if block_size > 0:
        dense = solution._apply_boat_rotation(dense, seed, block_size)
    gram = state.get("gram64")
    deployment_gram = state.get("deployment_gram")
    global_lrh = state.get("global_lrh")
    gram = gram if torch.is_tensor(gram) else None
    deployment_gram = deployment_gram if torch.is_tensor(deployment_gram) else None
    global_lrh = global_lrh if torch.is_tensor(global_lrh) else None
    gram = None if gram is None else gram.to(dense.device)
    deployment_gram = (
        None if deployment_gram is None else deployment_gram.to(dense.device)
    )
    global_lrh = None if global_lrh is None else global_lrh.to(dense.device)
    params = solution._dense_to_hif4(dense, offsets=solution._BASE_OFFSETS, gram64=gram)
    params = solution._refine_activation(dense, params, gram)
    _candidate, diagnostics = solution._refine_activation_global_lrh(
        dense,
        params,
        gram,
        deployment_gram,
        global_lrh,
        return_diagnostics=True,
    )
    return diagnostics


def run_diagnostic(
    cache_path: Path,
    solution_path: Path,
    layers: list[int],
    roles: list[str],
    mode: str,
) -> dict[str, Any]:
    started = time.perf_counter()
    cache = torch.load(cache_path, map_location="cpu", weights_only=False)
    solution = _load_solution(solution_path)
    total_layers = int(cache["layers"])
    selected_layers = sorted(set(int(layer) for layer in layers))
    if any(layer < 0 or layer >= total_layers for layer in selected_layers):
        raise ValueError(f"layer outside cache range 0..{total_layers - 1}")
    available_roles = set(str(role) for role in cache["roles"])
    unknown = [role for role in roles if role not in available_roles]
    if unknown:
        raise ValueError(f"unknown roles: {unknown}")

    records: list[dict[str, Any]] = []
    for layer in selected_layers:
        for role in roles:
            weight = cache["weights"][layer][role].to(torch.float32)
            weight_pair = nvfp4_encode(weight, mode)
            calibration_pairs = [
                nvfp4_encode(
                    cache["calibration_activations"][role][batch][layer], mode
                )
                for batch in range(len(cache["calibration_windows"]))
            ]
            calibrated = solution.hif4_calibration_and_quantize_weight(
                *weight_pair, calibration_pairs
            )
            state = calibrated["activation_state"]
            folds = []
            for batch, pair in enumerate(calibration_pairs):
                dense = solution._dequantize_nvfp4_float32(*pair)
                item = _fold_diagnostics(solution, dense, state)
                item["fold"] = batch
                folds.append(item)
            records.append({"layer": layer, "role": role, "folds": folds})

    proposal_rows = sum(
        fold["proposal_rows"]
        for record in records
        for fold in record["folds"]
    )
    accepted_rows = sum(
        fold["accepted_rows"]
        for record in records
        for fold in record["folds"]
    )
    conflict_rows = sum(
        fold["gram_mse_conflict_rows"]
        for record in records
        for fold in record["folds"]
    )
    fold_stable_cases = sum(
        int(
            len(record["folds"]) >= 2
            and all(fold["accepted_rows"] > 0 for fold in record["folds"])
        )
        for record in records
    )
    result = {
        "schema": 1,
        "diagnostic": "L3-global-activation-lrh-calibration-only",
        "cache": str(cache_path.resolve()),
        "solution": str(solution_path.resolve()),
        "mode": mode,
        "layers": selected_layers,
        "roles": roles,
        "sha256": {
            "cache_raw": hashlib.sha256(cache_path.read_bytes()).hexdigest(),
            "solution_lf": _sha256_lf(solution_path),
            "script_lf": _sha256_lf(Path(__file__).resolve()),
        },
        "records": records,
        "summary": {
            "case_count": len(records),
            "fold_count": sum(len(record["folds"]) for record in records),
            "proposal_rows": proposal_rows,
            "accepted_rows": accepted_rows,
            "gram_acceptance_rate": (
                accepted_rows / proposal_rows if proposal_rows else 0.0
            ),
            "gram_mse_conflict_rows": conflict_rows,
            "gram_mse_conflict_rate": (
                conflict_rows / proposal_rows if proposal_rows else 0.0
            ),
            "fold_stable_cases": fold_stable_cases,
            "fold_stability_rate": (
                fold_stable_cases / len(records) if records else 0.0
            ),
        },
        "elapsed_seconds": time.perf_counter() - started,
    }
    return result


def write_report(result: dict[str, Any], path: Path) -> None:
    summary = result["summary"]
    lines = [
        "# L3 Global Activation-LRH calibration diagnostic",
        "",
        "> Calibration-only; no test activation or output is read.",
        "",
        f"- Solution LF SHA256: `{result['sha256']['solution_lf']}`",
        f"- Layers: `{result['layers']}`; roles: `{', '.join(result['roles'])}`",
        f"- Elapsed: `{result['elapsed_seconds']:.3f}s`",
        "",
        "| metric | value |",
        "|---|---:|",
        f"| cases | {summary['case_count']} |",
        f"| folds | {summary['fold_count']} |",
        f"| proposal rows | {summary['proposal_rows']} |",
        f"| Gram accepted rows | {summary['accepted_rows']} |",
        f"| Gram acceptance rate | {summary['gram_acceptance_rate']:.6f} |",
        f"| Gram/MSE conflict rows | {summary['gram_mse_conflict_rows']} |",
        f"| Gram/MSE conflict rate | {summary['gram_mse_conflict_rate']:.6f} |",
        f"| fold-stable cases | {summary['fold_stable_cases']} |",
        f"| fold stability rate | {summary['fold_stability_rate']:.6f} |",
        "",
        "| layer | role | fold | proposals | Gram accepted | MSE accepted | conflicts |",
        "|---:|---|---:|---:|---:|---:|---:|",
    ]
    for record in result["records"]:
        for fold in record["folds"]:
            lines.append(
                f"| {record['layer']} | {record['role']} | {fold['fold']} | "
                f"{fold['proposal_rows']} | {fold['gram_accept_rows']} | "
                f"{fold['mse_accept_rows']} | {fold['gram_mse_conflict_rows']} |"
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--solution", type=Path, default=ROOT / "solution.py")
    parser.add_argument("--layers", type=int, nargs="+", default=list(DEFAULT_LAYERS))
    parser.add_argument("--roles", nargs="+", default=list(DEFAULT_ROLES))
    parser.add_argument("--mode", choices=("amax6", "amax4", "pow2"), default="amax6")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    result = run_diagnostic(args.cache, args.solution, args.layers, args.roles, args.mode)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_report(result, args.report)
    print(f"wrote {args.output}")
    print(f"wrote {args.report}")
    print(json.dumps(result["summary"], ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
