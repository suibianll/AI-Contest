"""Linear ceiling and error-decomposition dashboard (evaluator-side only).

This is the L0 diagnostic from the active HiF4 plan.  It evaluates the
current solution on a stratified set of layers and all Linear roles, but it
never changes ``solution.py`` or creates a deployment candidate.

For each layer/role it reports four evaluator-side output arms:

``both_player``
    The current deployed candidate (Q(A), Q(W)).
``weight_perfect``
    Candidate activation with the transformed dense weight.  This is the
    attainable arm if only the weight side became lossless.
``activation_perfect``
    Transformed dense activation with candidate weight.  This is the
    attainable arm if only the activation side became lossless.
``both_perfect``
    Both operands are dense in the candidate's equivalent frame.

It also computes a sampled legal scale-lattice oracle.  The oracle searches
all finite E6M2 scale codes while retaining HiF4's legal hierarchy and is
used only to measure local operand headroom.  It is never used to select an
online activation state.

Typical use::

    python evaluator/linear_ceiling_dashboard.py \
        --cache artifacts/real_model_suite/cache/qwen2.5-0.5b__seq128__calib2__test4__layersall__schema1.pt \
        --solution solution.py \
        --layers 0 5 11 17 23 \
        --output artifacts/oracle_dashboard/l0-linear-ceiling-qwen.json \
        --report logs/execution/2026-08-30-l0-linear-ceiling.md

The evaluator forms output products for diagnosis only.  No output product,
residual, test score, or official score is passed to a candidate API or
written to activation_state.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
import time
from pathlib import Path
from types import ModuleType
from typing import Any, Iterable

import torch

ROOT = Path(__file__).resolve().parents[1]
EVALUATOR_DIR = Path(__file__).resolve().parent
if str(EVALUATOR_DIR) not in sys.path:
    sys.path.insert(0, str(EVALUATOR_DIR))

from nvfp4_sim import nvfp4_encode  # noqa: E402
from real_data_eval import std_hif4  # noqa: E402
from reference_hif4 import dequantize_hif4, dequantize_nvfp4  # noqa: E402


BLOCK = 64
ALL_SCALE_OFFSETS = tuple(range(-254, 255))
DEFAULT_LAYERS = (0, 5, 11, 17, 23)
DEFAULT_ORACLE_ROWS = 32
DEFAULT_ROLES = ("q", "k", "v", "o", "fc_gate", "fc_up", "proj")
EPS = 1.0e-12


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_lf_text(path: Path) -> str:
    """Hash a text source after normalizing line endings to LF."""

    data = path.read_bytes().replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    return hashlib.sha256(data).hexdigest()


def _load_solution(path: Path) -> ModuleType:
    source = path.resolve()
    spec = importlib.util.spec_from_file_location("_hif4_l0_solution", source)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load solution: {source}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    required = (
        "hif4_calibration_and_quantize_weight",
        "hif4_dynamic_quantize_activation",
    )
    missing = [name for name in required if not callable(getattr(module, name, None))]
    if missing:
        raise AttributeError(f"solution is missing functions: {', '.join(missing)}")
    return module


def _state_frame(
    solution: ModuleType,
    state: Any,
    weight_reference: torch.Tensor,
    *,
    weight_side: bool,
) -> torch.Tensor:
    """Apply the exact frame helper used by the current candidate."""

    state = state if isinstance(state, dict) else {}
    smooth_inv = state.get("smooth_inv")
    if torch.is_tensor(smooth_inv):
        balance = smooth_inv.to(torch.float32).reciprocal().reshape(-1)
    else:
        balance = torch.ones(
            int(weight_reference.shape[-1]), dtype=torch.float32, device=weight_reference.device
        )
    permutation = state.get("permutation")
    if torch.is_tensor(permutation):
        permutation = permutation.to(dtype=torch.int64).reshape(-1)
    block_size = int(state.get("block_smooth_size", 0))
    seed = int(state.get("block_smooth_seed", -1))
    return solution._linear_pair_transform(
        weight_reference,
        balance,
        permutation,
        block_size,
        seed,
        weight_side=weight_side,
    )


def _block_loss(
    solution: ModuleType,
    dense: torch.Tensor,
    params: dict[str, torch.Tensor],
    gram64: torch.Tensor | None,
) -> torch.Tensor:
    rows, channels = map(int, dense.shape)
    if channels % BLOCK:
        raise ValueError(f"oracle requires channels divisible by {BLOCK}: {tuple(dense.shape)}")
    error = (
        solution._dequantize_hif4(params).to(torch.float32) - dense.to(torch.float32)
    ).reshape(rows, channels // BLOCK, BLOCK)
    if gram64 is None:
        return error.square().sum(dim=-1)
    return torch.einsum("rbi,bij,rbj->rb", error, gram64, error)


@torch.no_grad()
def _all_code_oracle_loss(
    solution: ModuleType,
    dense: torch.Tensor,
    gram64: torch.Tensor | None,
) -> torch.Tensor:
    """Return the best block loss over all 255 legal E6M2 scale codes.

    ``solution._encode_rows`` intentionally accepts an offset list and is
    excellent for the deployment path, but calling it once per offset is too
    expensive for a dashboard.  This routine vectorizes the 255-code sweep
    over a small row sample while using the exact same HiF4 hierarchy solver:
    each code still chooses the legal lv2/lv3 values and 3-bit mantissas.
    """

    dense = torch.nan_to_num(
        dense.detach().to(torch.float32),
        nan=0.0,
        posinf=float(solution._E6M2_MAX) * 7.0,
        neginf=-float(solution._E6M2_MAX) * 7.0,
    )
    rows, channels = map(int, dense.shape)
    blocks = channels // BLOCK
    grouped = dense.reshape(rows, blocks, 8, 2, 4)
    absolute = grouped.abs()
    sign = torch.sign(grouped)
    # [rows, blocks, codes]; code 0..254 is the complete finite scale lattice.
    codes = torch.arange(255, dtype=torch.int64, device=dense.device)
    scales = solution._e6m2_decode(codes).to(torch.float32)
    scales = scales.reshape(1, 1, 255)
    absolute_k = absolute.unsqueeze(2)
    losses: list[torch.Tensor] = []
    for exponent in (0, 1, 2):
        denominator = scales[..., None, None, None] * float(1 << exponent)
        mantissa = torch.round(
            absolute_k * (4.0 / denominator.clamp_min(EPS))
        ).clamp(0.0, 7.0) * 0.25
        losses.append(
            (absolute_k - mantissa * denominator).square().sum(dim=-1)
        )
    loss0, loss1, loss2 = losses
    choose01 = loss1 < loss0
    choose12 = loss2 < loss1
    cost1 = torch.minimum(loss0, loss1).sum(dim=-1)
    cost2 = torch.minimum(loss1, loss2).sum(dim=-1)
    use_lv2_two = cost2 < cost1
    use_lv3_two = torch.where(use_lv2_two.unsqueeze(-1), choose12, choose01)
    lv2 = 1.0 + use_lv2_two.to(torch.float32)
    lv3 = 1.0 + use_lv3_two.to(torch.float32)
    denominator = (
        scales[..., None, None, None]
        * lv2.unsqueeze(-1).unsqueeze(-1)
        * lv3.unsqueeze(-1)
    )
    mantissa = torch.round(
        absolute_k * (4.0 / denominator.clamp_min(EPS))
    ).clamp(0.0, 7.0) * 0.25
    error = (
        sign.unsqueeze(2) * mantissa * denominator - grouped.unsqueeze(2)
    ).reshape(rows, blocks, 255, BLOCK)
    if gram64 is None:
        losses_all = error.square().sum(dim=-1)
    else:
        gram = gram64.to(device=dense.device, dtype=torch.float32)
        losses_all = torch.einsum("rbki,bij,rbkj->rbk", error, gram, error)
    return losses_all.min(dim=2).values


@torch.no_grad()
def _scale_oracle(
    solution: ModuleType,
    dense: torch.Tensor,
    *,
    gram64: torch.Tensor | None,
    max_rows: int,
) -> dict[str, Any]:
    """Compare the active local scale set with all legal E6M2 codes."""

    dense = dense.detach().to(torch.float32)
    if dense.ndim != 2 or int(dense.shape[-1]) % BLOCK:
        return {"skipped": "not_2d_or_not_64_divisible", "shape": list(dense.shape)}
    sample = dense[: max(1, min(int(max_rows), int(dense.shape[0])))]
    baseline = solution._encode_rows(sample, solution._BASE_OFFSETS, gram64=gram64)
    baseline_loss = _block_loss(solution, sample, baseline, gram64)
    oracle_loss = _all_code_oracle_loss(solution, sample, gram64)
    base_total = float(baseline_loss.sum())
    oracle_total = float(oracle_loss.sum())
    improvement = max(base_total - oracle_total, 0.0)
    relative = (baseline_loss - oracle_loss).clamp_min(0.0) / baseline_loss.clamp_min(EPS)
    return {
        "shape": list(dense.shape),
        "sample_rows": int(sample.shape[0]),
        "sample_blocks": int(relative.numel()),
        "scale_candidates": 255,
        "local_offsets": list(solution._BASE_OFFSETS),
        "baseline_loss": base_total,
        "oracle_loss": oracle_total,
        "baseline_to_oracle_gap": improvement / max(base_total, EPS),
        "improved_blocks": int((baseline_loss - oracle_loss > 1.0e-8).sum()),
        "max_block_gap": float(relative.max()) if relative.numel() else 0.0,
    }


def _gain(standard_mse: float, player_mse: float) -> float:
    if standard_mse <= EPS:
        raise ZeroDivisionError("official case has non-positive MSE_STD")
    return (standard_mse - player_mse) / standard_mse


def _classify(weight_headroom: float, activation_headroom: float) -> str:
    max_headroom = max(weight_headroom, activation_headroom)
    if max_headroom < 1.0e-3:
        return "insufficient-headroom"
    ratio = (weight_headroom + 1.0e-9) / (activation_headroom + 1.0e-9)
    if 1.0 / 1.25 <= ratio <= 1.25:
        return "transform-coupled"
    return "weight-dominant" if weight_headroom > activation_headroom else "activation-dominant"


@torch.no_grad()
def _run_case(
    solution: ModuleType,
    weight_pair: tuple[torch.Tensor, torch.Tensor],
    activation_pairs: list[tuple[torch.Tensor, torch.Tensor]],
    state: Any,
    weight_params: dict[str, torch.Tensor],
) -> dict[str, Any]:
    weight_reference = dequantize_nvfp4(*weight_pair).to(torch.float32)
    weight_standard = std_hif4(solution, weight_reference)
    weight_player = dequantize_hif4(weight_params, weight_reference.shape).to(torch.float32)
    weight_reference_frame = _state_frame(
        solution, state, weight_reference, weight_side=True
    )
    arms: dict[str, list[float]] = {
        "both_player": [],
        "weight_perfect": [],
        "activation_perfect": [],
        "both_perfect": [],
    }
    standard_mses: list[float] = []
    frame_errors: list[float] = []
    for activation_pair in activation_pairs:
        activation_reference = dequantize_nvfp4(*activation_pair).to(torch.float32)
        activation_standard = std_hif4(solution, activation_reference)
        reference = activation_reference @ weight_reference.T
        standard = activation_standard @ weight_standard.T
        player_params = solution.hif4_dynamic_quantize_activation(
            *activation_pair, state
        )
        player_activation = dequantize_hif4(
            player_params, activation_reference.shape
        ).to(torch.float32)
        activation_reference_frame = _state_frame(
            solution, state, activation_reference, weight_side=False
        )
        frame_reference = activation_reference_frame @ weight_reference_frame.T
        frame_errors.append(float((frame_reference - reference).square().mean()))
        outputs = {
            "both_player": player_activation @ weight_player.T,
            "weight_perfect": player_activation @ weight_reference_frame.T,
            "activation_perfect": activation_reference_frame @ weight_player.T,
            "both_perfect": frame_reference,
        }
        standard_mse = float((standard - reference).square().mean())
        standard_mses.append(standard_mse)
        for name, output in outputs.items():
            arms[name].append(_gain(standard_mse, float((output - reference).square().mean())))
    means = {name: sum(values) / len(values) for name, values in arms.items()}
    both_player = means["both_player"]
    return {
        "case_count": len(activation_pairs),
        "arm_gain_mean": means,
        "weight_side_headroom": means["weight_perfect"] - both_player,
        "activation_side_headroom": means["activation_perfect"] - both_player,
        "relaxed_both_perfect_headroom": means["both_perfect"] - both_player,
        "standard_mse_mean": sum(standard_mses) / len(standard_mses),
        "frame_reference_mse_mean": sum(frame_errors) / len(frame_errors),
        "classification": _classify(
            means["weight_perfect"] - both_player,
            means["activation_perfect"] - both_player,
        ),
    }


def _aggregate(items: Iterable[dict[str, Any]]) -> dict[str, Any]:
    values = list(items)
    if not values:
        return {}
    arm_names = ("both_player", "weight_perfect", "activation_perfect", "both_perfect")
    arms = {
        name: sum(float(item["arm_gain_mean"][name]) for item in values) / len(values)
        for name in arm_names
    }
    weight_headroom = sum(float(item["weight_side_headroom"]) for item in values) / len(values)
    activation_headroom = sum(float(item["activation_side_headroom"]) for item in values) / len(values)
    return {
        "count": len(values),
        "arm_gain_mean": arms,
        "weight_side_headroom": weight_headroom,
        "activation_side_headroom": activation_headroom,
        "relaxed_both_perfect_headroom": (
            sum(float(item["relaxed_both_perfect_headroom"]) for item in values) / len(values)
        ),
        "classification": _classify(weight_headroom, activation_headroom),
    }


@torch.no_grad()
def run_dashboard(
    cache_path: Path,
    solution_path: Path,
    layers: Iterable[int],
    *,
    roles: Iterable[str] = DEFAULT_ROLES,
    mode: str = "amax6",
    oracle_rows: int = DEFAULT_ORACLE_ROWS,
) -> dict[str, Any]:
    started = time.perf_counter()
    cache = torch.load(cache_path, map_location="cpu", weights_only=False)
    solution = _load_solution(solution_path)
    all_layers = int(cache["layers"])
    selected_layers = sorted(set(int(layer) for layer in layers))
    invalid_layers = [layer for layer in selected_layers if layer < 0 or layer >= all_layers]
    if invalid_layers:
        raise ValueError(f"layers outside cache range 0..{all_layers - 1}: {invalid_layers}")
    available_roles = tuple(str(role) for role in cache["roles"])
    selected_roles = tuple(str(role) for role in roles)
    unknown_roles = [role for role in selected_roles if role not in available_roles]
    if unknown_roles:
        raise ValueError(f"roles not present in cache: {unknown_roles}")

    records: list[dict[str, Any]] = []
    for layer in selected_layers:
        for role in selected_roles:
            weight_dense = cache["weights"][layer][role].to(torch.float32)
            calibration_pairs = [
                nvfp4_encode(cache["calibration_activations"][role][batch][layer], mode)
                for batch in range(len(cache["calibration_windows"]))
            ]
            calibrated = solution.hif4_calibration_and_quantize_weight(
                *nvfp4_encode(weight_dense, mode), calibration_pairs
            )
            if not isinstance(calibrated, dict):
                raise TypeError("weight calibration did not return a dict")
            state = calibrated["activation_state"]
            weight_params = calibrated["weight_params"]
            weight_reference = dequantize_nvfp4(
                *nvfp4_encode(weight_dense, mode)
            ).to(torch.float32)
            weight_frame = _state_frame(solution, state, weight_reference, weight_side=True)
            gram64 = solution._gram64(weight_frame)
            activation_frames = [
                _state_frame(
                    solution,
                    state,
                    dequantize_nvfp4(
                        *nvfp4_encode(
                            cache["calibration_activations"][role][batch][layer], mode
                        )
                    ).to(torch.float32),
                    weight_side=False,
                )
                for batch in range(len(cache["calibration_windows"]))
            ]
            oracle = {
                "weight_plain": _scale_oracle(
                    solution,
                    weight_frame,
                    gram64=None,
                    max_rows=oracle_rows,
                ),
                "weight_gram": _scale_oracle(
                    solution,
                    weight_frame,
                    gram64=gram64,
                    max_rows=oracle_rows,
                ),
                "activation_gram": _aggregate_oracles(
                    [
                        _scale_oracle(
                            solution,
                            activation_frame,
                            gram64=gram64,
                            max_rows=oracle_rows,
                        )
                        for activation_frame in activation_frames
                    ]
                ),
            }
            test_pairs = [
                nvfp4_encode(cache["test_activations"][role][batch][layer], mode)
                for batch in range(len(cache["test_windows"]))
            ]
            arms = _run_case(solution, nvfp4_encode(weight_dense, mode), test_pairs, state, weight_params)
            state_summary = {
                "block_smooth_size": int(state.get("block_smooth_size", 0)) if isinstance(state, dict) else 0,
                "block_smooth_seed": int(state.get("block_smooth_seed", -1)) if isinstance(state, dict) else -1,
            }
            smooth_inv = state.get("smooth_inv") if isinstance(state, dict) else None
            if torch.is_tensor(smooth_inv):
                balance = smooth_inv.to(torch.float32).reciprocal()
                state_summary.update(
                    {
                        "balance_min": float(balance.min()),
                        "balance_max": float(balance.max()),
                        "balance_condition": float(balance.max() / balance.clamp_min(EPS).min()),
                    }
                )
            records.append(
                {
                    "layer": layer,
                    "role": role,
                    "weight_shape": list(weight_dense.shape),
                    "state": state_summary,
                    "arms": arms,
                    "oracle": oracle,
                }
            )

    by_layer = {
        str(layer): _aggregate(item["arms"] for item in records if item["layer"] == layer)
        for layer in selected_layers
    }
    by_role = {
        role: _aggregate(item["arms"] for item in records if item["role"] == role)
        for role in selected_roles
    }
    overall = _aggregate(item["arms"] for item in records)
    elapsed = time.perf_counter() - started
    result = {
        "schema": 1,
        "diagnostic": "L0-linear-ceiling-dashboard",
        "compliance": {
            "deployment_path_modified": False,
            "candidate_state_written": False,
            "test_output_used_for_selection": False,
            "official_score_used": False,
            "oracle_is_diagnostic_only": True,
        },
        "cache": str(cache_path.resolve()),
        "solution": str(solution_path.resolve()),
        "mode": mode,
        "layers": selected_layers,
        "roles": list(selected_roles),
        "oracle_rows": int(oracle_rows),
        "oracle_scale_codes": 255,
        "sha256": {
            "cache": _sha256_file(cache_path),
            "solution_lf": _sha256_lf_text(solution_path),
            "dashboard_script_lf": _sha256_lf_text(Path(__file__).resolve()),
        },
        "cache_metadata": {
            "model": cache.get("metadata", {}).get("model"),
            "layers": all_layers,
            "hidden_size": cache.get("hidden_size"),
            "calibration_samples": len(cache.get("calibration_windows", [])),
            "test_samples": len(cache.get("test_windows", [])),
        },
        "definition": {
            "gain": "(MSE_STD-MSE_PLAYER)/MSE_STD",
            "weight_perfect": "Q(A) with transformed dense W",
            "activation_perfect": "transformed dense A with Q(W)",
            "both_perfect": "transformed dense A and W",
            "weight_headroom": "gain(weight_perfect)-gain(both_player)",
            "activation_headroom": "gain(activation_perfect)-gain(both_player)",
            "oracle": "sampled all-255 E6M2 scale codes with legal lv2/lv3 hierarchy",
        },
        "records": records,
        "by_layer": by_layer,
        "by_role": by_role,
        "overall": overall,
        "elapsed_seconds": elapsed,
    }
    return result


def _aggregate_oracles(items: Iterable[dict[str, Any]]) -> dict[str, Any]:
    values = [item for item in items if "skipped" not in item]
    if not values:
        return {"skipped": "no_valid_samples"}
    numeric = ("baseline_loss", "oracle_loss", "baseline_to_oracle_gap", "max_block_gap")
    result: dict[str, Any] = {
        "sample_count": len(values),
        "sample_rows": sum(int(item["sample_rows"]) for item in values),
        "sample_blocks": sum(int(item["sample_blocks"]) for item in values),
        "improved_blocks": sum(int(item["improved_blocks"]) for item in values),
    }
    for key in numeric:
        result[key] = sum(float(item[key]) for item in values) / len(values)
    return result


def write_report(result: dict[str, Any], path: Path) -> None:
    lines = [
        "# L0 Linear ceiling / error decomposition",
        "",
        "> evaluator-side diagnostic only; no deployment code or activation state was changed.",
        "",
        f"- Cache: `{result['cache']}`",
        f"- Solution: `{result['solution']}`",
        f"- Layers: `{result['layers']}`; roles: `{', '.join(result['roles'])}`",
        f"- Oracle rows per sample: `{result['oracle_rows']}`; scale candidates: `255`",
        f"- Solution LF SHA256: `{result['sha256']['solution_lf']}`",
        f"- Dashboard LF SHA256: `{result['sha256']['dashboard_script_lf']}`",
        f"- Elapsed: `{result['elapsed_seconds']:.3f}s`",
        "",
        "## Overall deployment arms",
        "",
        "| arm | mean gain |",
        "|---|---:|",
    ]
    for name, value in result["overall"]["arm_gain_mean"].items():
        lines.append(f"| {name} | `{value:.8f}` |")
    lines.extend(
        [
            "",
            f"Weight-side headroom: `{result['overall']['weight_side_headroom']:.8f}`; "
            f"activation-side headroom: `{result['overall']['activation_side_headroom']:.8f}`; "
            f"relaxed both-perfect headroom: `{result['overall']['relaxed_both_perfect_headroom']:.8f}`.",
            "",
            f"Diagnostic classification: **{result['overall']['classification']}**.",
            "",
            "## Layer summary",
            "",
            "| layer | both player | weight perfect | activation perfect | both perfect | W headroom | A headroom | class |",
            "|---:|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    for layer, item in result["by_layer"].items():
        arms = item["arm_gain_mean"]
        lines.append(
            f"| {layer} | {arms['both_player']:.8f} | {arms['weight_perfect']:.8f} | "
            f"{arms['activation_perfect']:.8f} | {arms['both_perfect']:.8f} | "
            f"{item['weight_side_headroom']:.8f} | {item['activation_side_headroom']:.8f} | "
            f"{item['classification']} |"
        )
    lines.extend(
        [
            "",
            "## Role summary",
            "",
            "| role | both player | weight perfect | activation perfect | both perfect | W headroom | A headroom | class |",
            "|---|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    for role, item in result["by_role"].items():
        arms = item["arm_gain_mean"]
        lines.append(
            f"| {role} | {arms['both_player']:.8f} | {arms['weight_perfect']:.8f} | "
            f"{arms['activation_perfect']:.8f} | {arms['both_perfect']:.8f} | "
            f"{item['weight_side_headroom']:.8f} | {item['activation_side_headroom']:.8f} | "
            f"{item['classification']} |"
        )
    lines.extend(
        [
            "",
            "## Legal scale oracle summary",
            "",
            "The oracle searches all finite E6M2 scale codes while retaining the legal HiF4 hierarchy. "
            "It is a sampled operand-local ceiling diagnostic, not a deployment candidate.",
            "",
            "| layer | role | weight plain gap | weight Gram gap | activation Gram gap |",
            "|---:|---|---:|---:|---:|",
        ]
    )
    for record in result["records"]:
        oracle = record["oracle"]
        activation = oracle["activation_gram"]
        lines.append(
            f"| {record['layer']} | {record['role']} | "
            f"{oracle['weight_plain'].get('baseline_to_oracle_gap', float('nan')):.8f} | "
            f"{oracle['weight_gram'].get('baseline_to_oracle_gap', float('nan')):.8f} | "
            f"{activation.get('baseline_to_oracle_gap', float('nan')):.8f} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            "1. `weight_perfect` and `activation_perfect` are evaluator-side one-sided arms; they do not claim that a legal algorithm can reach those values.",
            "2. The 255-code oracle uses calibration tensors and static Gram only. It never writes a state or selects a test-time candidate.",
            "3. A small scale-oracle gap rules out scale search as the main source of a large gain, but does not rule out coordinate transforms or cross-block solvers.",
            "4. A large one-sided arm is headroom evidence, not a guarantee of cross-layer transfer. L1/L2 still require the stratified and full-layer gates in the active plan.",
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
    parser.add_argument("--oracle-rows", type=int, default=DEFAULT_ORACLE_ROWS)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if args.oracle_rows <= 0:
        raise ValueError("--oracle-rows must be positive")
    result = run_dashboard(
        args.cache,
        args.solution,
        args.layers,
        roles=args.roles,
        mode=args.mode,
        oracle_rows=args.oracle_rows,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_report(result, args.report)
    print(f"wrote {args.output}")
    print(f"wrote {args.report}")
    print(
        "overall "
        f"both={result['overall']['arm_gain_mean']['both_player']:.8f} "
        f"Wperfect={result['overall']['arm_gain_mean']['weight_perfect']:.8f} "
        f"Aperfect={result['overall']['arm_gain_mean']['activation_perfect']:.8f} "
        f"class={result['overall']['classification']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
