"""Fast stratified Linear screen for a candidate solution.

This evaluator-side tool is used by L1 after the L0 ceiling dashboard.  It
reuses the exact Linear arms but skips the 255-code oracle, so a candidate can
be screened on sparse layers before a full 24-layer run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
EVALUATOR_DIR = Path(__file__).resolve().parent
if str(EVALUATOR_DIR) not in sys.path:
    sys.path.insert(0, str(EVALUATOR_DIR))

from linear_ceiling_dashboard import _aggregate, _load_solution, _run_case  # noqa: E402
from nvfp4_sim import nvfp4_encode  # noqa: E402


DEFAULT_LAYERS = (0, 5, 11, 17, 23)
DEFAULT_ROLES = ("q", "k", "v", "o", "fc_gate", "fc_up", "proj")


def _sha256_lf(path: Path) -> str:
    data = path.read_bytes().replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    return hashlib.sha256(data).hexdigest()


def run_screen(
    cache_path: Path,
    solution_path: Path,
    layers: list[int],
    roles: list[str],
    mode: str,
    stage: str = "candidate",
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
            test_pairs = [
                nvfp4_encode(cache["test_activations"][role][batch][layer], mode)
                for batch in range(len(cache["test_windows"]))
            ]
            arms = _run_case(
                solution,
                weight_pair,
                test_pairs,
                calibrated["activation_state"],
                calibrated["weight_params"],
            )
            records.append({"layer": layer, "role": role, "arms": arms})

    by_layer = {
        str(layer): _aggregate(item["arms"] for item in records if item["layer"] == layer)
        for layer in selected_layers
    }
    by_role = {
        role: _aggregate(item["arms"] for item in records if item["role"] == role)
        for role in roles
    }
    result = {
        "schema": 1,
        "diagnostic": f"{stage}-stratified-linear-candidate-screen",
        "cache": str(cache_path.resolve()),
        "solution": str(solution_path.resolve()),
        "mode": mode,
        "layers": selected_layers,
        "roles": roles,
        "sha256": {
            "cache_raw": hashlib.sha256(cache_path.read_bytes()).hexdigest(),
            "solution_lf": _sha256_lf(solution_path),
            "screen_script_lf": _sha256_lf(Path(__file__).resolve()),
        },
        "records": records,
        "by_layer": by_layer,
        "by_role": by_role,
        "overall": _aggregate(item["arms"] for item in records),
        "elapsed_seconds": time.perf_counter() - started,
    }
    return result


def write_report(result: dict[str, Any], path: Path) -> None:
    stage = str(result["diagnostic"]).split("-", 1)[0]
    lines = [
        f"# {stage} stratified Linear candidate screen",
        "",
        "> evaluator-side screen; no test output is used to select online state.",
        "",
        f"- Solution: `{result['solution']}`",
        f"- Layers: `{result['layers']}`; roles: `{', '.join(result['roles'])}`",
        f"- Solution LF SHA256: `{result['sha256']['solution_lf']}`",
        f"- Elapsed: `{result['elapsed_seconds']:.3f}s`",
        "",
        "| layer | both player | weight perfect | activation perfect |",
        "|---:|---:|---:|---:|",
    ]
    for layer, item in result["by_layer"].items():
        arms = item["arm_gain_mean"]
        lines.append(
            f"| {layer} | {arms['both_player']:.8f} | "
            f"{arms['weight_perfect']:.8f} | {arms['activation_perfect']:.8f} |"
        )
    lines.extend(
        [
            "",
            "| role | both player | weight perfect | activation perfect |",
            "|---|---:|---:|---:|",
        ]
    )
    for role, item in result["by_role"].items():
        arms = item["arm_gain_mean"]
        lines.append(
            f"| {role} | {arms['both_player']:.8f} | "
            f"{arms['weight_perfect']:.8f} | {arms['activation_perfect']:.8f} |"
        )
    lines.extend(
        [
            "",
            f"Overall selected-layer Linear mean: `{result['overall']['arm_gain_mean']['both_player']:.8f}`.",
            "This is a stratified screen, not the 24-layer parent gate.",
        ]
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
    parser.add_argument("--stage", default="candidate")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    result = run_screen(
        args.cache, args.solution, args.layers, args.roles, args.mode, args.stage
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_report(result, args.report)
    print(f"wrote {args.output}")
    print(f"wrote {args.report}")
    print(f"overall={result['overall']['arm_gain_mean']['both_player']:.8f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
