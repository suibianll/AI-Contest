"""Fast cross-fold stability probe for the L3 fc legal-code teacher.

This is a research-only diagnostic.  It reuses the immutable pre-A3 parent and
the bounded joint teacher, but restricts the run to layer 3 and the two fc
roles.  The probe records per-block cheap features and asks whether a rule
chosen on calibration fold 0 can predict the sign of the exact output margin
on fold 1 (and vice versa).  No rule is written into an API state.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

ROOT = Path(__file__).resolve().parents[1]
EVALUATOR_DIR = ROOT / "evaluator"
if str(EVALUATOR_DIR) not in sys.path:
    sys.path.insert(0, str(EVALUATOR_DIR))

import official_eval as evaluator  # noqa: E402


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load source: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


ORACLE = _load_module(
    "_l3_fc_legal_oracle_stability",
    ROOT / "workbench" / "l3_fc_legal_oracle.py",
)
FAST = _load_module(
    "_l3_fc_fast_probe_stability",
    ROOT / "workbench" / "l3_fc_fast_probe.py",
)
PARENT_HASH = "800ca10ec3414e4fe886b93ca62bd4a350d26bba015287df7e8df2dd871ac23d"
LAYERS = (3,)
ROLES = ("fc_gate", "fc_up")
FOLDS = (0, 1)
BLOCK = 64
EPS = 1.0e-12


def _load_parent(path: Path):
    return ORACLE._load_parent(path)


def _finite(value: float) -> float:
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"non-finite value: {value}")
    return value


def _corr(left: Sequence[float], right: Sequence[float]) -> float | None:
    if len(left) != len(right) or len(left) < 2:
        return None
    x = torch.tensor(left, dtype=torch.float64)
    y = torch.tensor(right, dtype=torch.float64)
    x = x - x.mean()
    y = y - y.mean()
    denom = torch.sqrt((x.square().sum()) * (y.square().sum()))
    if float(denom) <= 1.0e-15:
        return None
    return _finite(float((x * y).sum().div(denom).item()))


def _rank(values: Sequence[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda index: (float(values[index]), index))
    result = [0.0] * len(values)
    for position, index in enumerate(order):
        result[index] = float(position)
    return result


def _sign(value: float) -> int:
    return 1 if value > 0.0 else -1 if value < 0.0 else 0


def _binary_metrics(predicted: Sequence[bool], actual: Sequence[bool]) -> dict[str, Any]:
    if len(predicted) != len(actual):
        raise ValueError("prediction/label length mismatch")
    tp = sum(bool(p) and bool(a) for p, a in zip(predicted, actual))
    fp = sum(bool(p) and not bool(a) for p, a in zip(predicted, actual))
    fn = sum((not bool(p)) and bool(a) for p, a in zip(predicted, actual))
    tn = sum((not bool(p)) and (not bool(a)) for p, a in zip(predicted, actual))
    return {
        "count": len(actual),
        "predicted_positive": int(sum(bool(value) for value in predicted)),
        "actual_positive": int(sum(bool(value) for value in actual)),
        "true_positive": int(tp),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_negative": int(tn),
        "accuracy": _finite((tp + tn) / max(len(actual), 1)),
        "precision": None if tp + fp == 0 else _finite(tp / (tp + fp)),
        "recall": None if tp + fn == 0 else _finite(tp / (tp + fn)),
    }


def _features_for_block(
    target: torch.Tensor,
    parent_error: torch.Tensor,
    gram: torch.Tensor,
    q_block: torch.Tensor,
    denominator: torch.Tensor,
    scale: torch.Tensor,
    lv2: torch.Tensor,
    lv3: torch.Tensor,
) -> dict[str, float]:
    """Return only static, state-local features suitable for a compiler."""
    values = q_block.detach().to(torch.float32)
    target = target.detach().to(torch.float32)
    parent_error = parent_error.detach().to(torch.float32)
    code = torch.round(values * 4.0 / denominator.clamp_min(EPS)).clamp(-7.0, 7.0)
    scale_code = ORACLE.evaluator_e6_code(scale).to(torch.float32)
    return {
        "target_amax": _finite(float(target.abs().amax(dim=-1).mean().item())),
        "target_rms": _finite(float(target.square().mean(dim=-1).sqrt().mean().item())),
        "parent_error_rms": _finite(float(parent_error.square().mean(dim=-1).sqrt().mean().item())),
        "parent_quadratic_loss": _finite(float(torch.einsum("ri,ij,rj->r", parent_error, gram, parent_error).mean().item())),
        "code_abs_mean": _finite(float(code.abs().mean().item())),
        "code_zero_fraction": _finite(float((code == 0.0).to(torch.float32).mean().item())),
        "lv2_high_fraction": _finite(float((lv2 > 1.0).to(torch.float32).mean().item())),
        "lv3_high_fraction": _finite(float((lv3 > 1.0).to(torch.float32).mean().item())),
        "scale_code": _finite(float(scale_code.mean().item())),
    }


def _run_record(parent: Any, raw: Any, role: str, device: torch.device) -> list[dict[str, Any]]:
    layer = 3
    weight_pair = evaluator._pair(raw.weights[layer][role])
    weight_pair = (weight_pair[0].to(device), weight_pair[1].to(device))
    calibration_pairs = []
    for fold in FOLDS:
        pair = evaluator._pair(raw.calibration_activations[role][fold][layer])
        calibration_pairs.append((pair[0].to(device), pair[1].to(device)))

    calibration = parent.hif4_calibration_and_quantize_weight(
        weight_pair[0], weight_pair[1], calibration_pairs
    )
    evaluator.validate_state(calibration["activation_state"])
    evaluator.validate_hif4_params(
        calibration["weight_params"], evaluator.dequantize_nvfp4(*weight_pair).shape
    )
    weight_t, activation_t = parent._active_transformed_calibration(
        weight_pair[0], weight_pair[1], calibration_pairs, calibration["activation_state"]
    )
    weight_t = weight_t.to(torch.float32)
    q_weight = parent._dequantize_hif4(calibration["weight_params"]).to(torch.float32)
    state = calibration["activation_state"]
    gram = state.get("gram64")
    cross = state.get("output_cross64")
    if not torch.is_tensor(gram) or not torch.is_tensor(cross):
        raise ValueError("parent state lacks deployed gram/cross")
    gram = gram.to(device=device, dtype=torch.float32)
    cross = cross.to(device=device, dtype=torch.float32)

    output: list[dict[str, Any]] = []
    for fold, calibration_pair in enumerate(calibration_pairs):
        incumbent = parent.hif4_dynamic_quantize_activation(
            calibration_pair[0], calibration_pair[1], state
        )
        evaluator.validate_hif4_params(
            incumbent, evaluator.dequantize_nvfp4(*calibration_pair).shape
        )
        q_dense = parent._dequantize_hif4(incumbent).to(torch.float32)
        target = activation_t[fold].to(torch.float32)
        rows, channels = map(int, q_dense.shape)
        blocks = channels // BLOCK
        params = {name: value.to(device=device) for name, value in incumbent.items()}
        scale_all = params["scale_factor"].reshape(rows, blocks)
        lv2_all = params["scale_lv2"].reshape(rows, blocks, 8)
        lv3_all = params["scale_lv3"].reshape(rows, blocks, 8, 2)
        q_all = q_dense.reshape(rows, blocks, BLOCK)
        target_blocks = target.reshape(rows, blocks, BLOCK)
        parent_error = q_all - target_blocks
        target_output = target @ weight_t.transpose(0, 1)
        parent_output = q_dense @ q_weight.transpose(0, 1)
        parent_residual = parent_output - target_output
        parent_row_loss = parent_residual.square().mean(dim=1)

        teacher_q = q_all.clone()
        teacher_scale = scale_all.clone()
        teacher_lv2 = lv2_all.clone()
        teacher_lv3 = lv3_all.clone()
        block_features: list[dict[str, float]] = []
        quadratic_margins: list[float] = []
        for block in range(blocks):
            denominator = ORACLE._denominator(
                scale_all[:, block], lv2_all[:, block], lv3_all[:, block]
            )
            block_features.append(
                _features_for_block(
                    target_blocks[:, block], parent_error[:, block], gram[block],
                    q_all[:, block], denominator, scale_all[:, block],
                    lv2_all[:, block], lv3_all[:, block],
                )
            )
            result = ORACLE._run_teacher(
                "joint",
                q_all[:, block],
                scale_all[:, block],
                lv2_all[:, block],
                lv3_all[:, block],
                target_blocks[:, block],
                gram[block],
                cross[block],
                parent,
            )
            teacher_q[:, block], teacher_scale[:, block], teacher_lv2[:, block], teacher_lv3[:, block], _ = result
            quadratic_margins.append(
                _finite(float(ORACLE._block_quadratic(
                    q_all[:, block], gram[block], cross[block], target_blocks[:, block]
                ).sub(ORACLE._block_quadratic(
                    teacher_q[:, block], gram[block], cross[block], target_blocks[:, block]
                )).div(
                    ORACLE._block_quadratic(q_all[:, block], gram[block], cross[block], target_blocks[:, block]).abs().clamp_min(EPS)
                ).mean().item()))
            )

        exact_margins: list[float] = []
        exact_parent = float(parent_residual.square().mean().item())
        for block in range(blocks):
            start, stop = block * BLOCK, (block + 1) * BLOCK
            delta = (teacher_q[:, block] - q_all[:, block]) @ q_weight[:, start:stop].transpose(0, 1)
            single_loss = (parent_residual + delta).square().mean(dim=1)
            exact_margins.append(
                _finite(float((parent_row_loss.mean() - single_loss.mean()).div(parent_row_loss.mean().clamp_min(EPS)).item()))
            )
        teacher_output = teacher_q.reshape(rows, channels) @ q_weight.transpose(0, 1)
        exact_teacher = float((teacher_output - target_output).square().mean().item())
        output.append({
            "layer": layer,
            "role": role,
            "fold": fold,
            "calibration_length": len(raw.calibration_windows[fold].input_ids),
            "rows": rows,
            "channels": channels,
            "blocks": [
                {
                    "block": block,
                    "features": block_features[block],
                    "quadratic_margin": quadratic_margins[block],
                    "exact_single_block_margin": exact_margins[block],
                    "exact_single_sign": _sign(exact_margins[block]),
                }
                for block in range(blocks)
            ],
            "parent_exact_output_mse": _finite(exact_parent),
            "teacher_exact_output_mse": _finite(exact_teacher),
            "teacher_joint_margin": _finite((exact_parent - exact_teacher) / max(exact_parent, EPS)),
        })
    return output


def _rule_report(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by_role: dict[str, dict[int, Mapping[str, Any]]] = {}
    for record in records:
        by_role.setdefault(str(record["role"]), {})[int(record["fold"])] = record
    feature_names = tuple(records[0]["blocks"][0]["features"].keys())
    result: dict[str, Any] = {"by_role": {}, "all": {}}
    for role, folds in by_role.items():
        left = {int(item["block"]): item for item in folds[0]["blocks"]}
        right = {int(item["block"]): item for item in folds[1]["blocks"]}
        common = sorted(set(left) & set(right))
        role_result: dict[str, Any] = {"block_count": len(common), "features": {}, "rules": []}
        left_margins = [float(left[index]["exact_single_block_margin"]) for index in common]
        right_margins = [float(right[index]["exact_single_block_margin"]) for index in common]
        sign_pairs = [(_sign(left[index]["exact_single_block_margin"]), _sign(right[index]["exact_single_block_margin"])) for index in common]
        role_result["decision_sign_agreement"] = int(sum(a == b for a, b in sign_pairs))
        role_result["decision_sign_pairs"] = [{"block": index, "fold0": a, "fold1": b} for index, (a, b) in zip(common, sign_pairs)]
        role_result["exact_margin_correlation"] = _corr(left_margins, right_margins)
        for feature in feature_names:
            left_values = [float(left[index]["features"][feature]) for index in common]
            right_values = [float(right[index]["features"][feature]) for index in common]
            role_result["features"][feature] = {
                "pearson": _corr(left_values, right_values),
                "spearman": _corr(_rank(left_values), _rank(right_values)),
            }
            thresholds = sorted(set([
                statistics.quantiles(left_values, n=4, method="inclusive")[0],
                statistics.median(left_values),
                statistics.quantiles(left_values, n=4, method="inclusive")[2],
            ])) if len(left_values) >= 2 else [left_values[0]]
            actual_left = [value > 0.0 for value in left_margins]
            actual_right = [value > 0.0 for value in right_margins]
            for direction in ("ge", "le"):
                for threshold in thresholds:
                    pred_left = [value >= threshold if direction == "ge" else value <= threshold for value in left_values]
                    pred_right = [value >= threshold if direction == "ge" else value <= threshold for value in right_values]
                    role_result["rules"].append({
                        "feature": feature,
                        "direction": direction,
                        "threshold": _finite(threshold),
                        "train_fold": 0,
                        "train": _binary_metrics(pred_left, actual_left),
                        "heldout_fold": 1,
                        "heldout": _binary_metrics(pred_right, actual_right),
                    })
        best = sorted(
            role_result["rules"],
            key=lambda item: (
                -float(item["train"]["accuracy"]),
                -float(item["heldout"]["accuracy"]),
                str(item["feature"]),
                str(item["direction"]),
                float(item["threshold"]),
            ),
        )
        role_result["best_train_rule"] = best[0] if best else None
        result["by_role"][role] = role_result

    # Repeat in the opposite direction so the report cannot hide a fold-specific fit.
    reverse: list[dict[str, Any]] = []
    for role, folds in by_role.items():
        left = {int(item["block"]): item for item in folds[0]["blocks"]}
        right = {int(item["block"]): item for item in folds[1]["blocks"]}
        common = sorted(set(left) & set(right))
        for feature in feature_names:
            right_values = [float(right[index]["features"][feature]) for index in common]
            thresholds = sorted(set([
                statistics.quantiles(right_values, n=4, method="inclusive")[0],
                statistics.median(right_values),
                statistics.quantiles(right_values, n=4, method="inclusive")[2],
            ])) if len(right_values) >= 2 else [right_values[0]]
            actual_right = [float(right[index]["exact_single_block_margin"]) > 0.0 for index in common]
            actual_left = [float(left[index]["exact_single_block_margin"]) > 0.0 for index in common]
            for direction in ("ge", "le"):
                for threshold in thresholds:
                    pred_right = [value >= threshold if direction == "ge" else value <= threshold for value in right_values]
                    left_values = [float(left[index]["features"][feature]) for index in common]
                    pred_left = [value >= threshold if direction == "ge" else value <= threshold for value in left_values]
                    reverse.append({
                        "role": role,
                        "feature": feature,
                        "direction": direction,
                        "threshold_from_fold": 1,
                        "threshold": _finite(threshold),
                        "train": _binary_metrics(pred_right, actual_right),
                        "heldout": _binary_metrics(pred_left, actual_left),
                    })
    result["reverse_rules"] = reverse
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--parent", type=Path,
        default=ROOT / "workbench" / "pre-a3-v147-parent.py",
    )
    parser.add_argument(
        "--cache", type=Path,
        default=ROOT / "artifacts" / "official_eval" / "cache" / "qwen2.5-0.5b-proxy-v2.pt",
    )
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "artifacts" / "official_eval" / "l3-fc-stability-probe.json",
    )
    parser.add_argument(
        "--report", type=Path,
        default=ROOT / "logs" / "execution" / "2026-09-02-l3-fc-stability-probe.md",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    parent_path = args.parent.resolve()
    parent_hash = hashlib.sha256(parent_path.read_bytes()).hexdigest()
    if parent_hash.lower() != PARENT_HASH:
        raise ValueError(f"unexpected parent SHA {parent_hash}; expected {PARENT_HASH}")
    device = torch.device(args.device)

    # Use the same one-pass Jacobi approximation as the already-recorded fast
    # layer-3 screen; canonical D0 remains the source of teacher upper bounds.
    ORACLE.DEFAULT_LAYERS = LAYERS
    ORACLE.DEFAULT_ROLES = ROLES
    ORACLE.EDIT_CLASSES = ("joint",)
    ORACLE._coordinate_pass = FAST._coordinate_pass_jacobi

    started = time.perf_counter()
    parent = _load_parent(parent_path)
    raw = evaluator.load_pack(args.cache.resolve())
    records: list[dict[str, Any]] = []
    for role in ROLES:
        print(f"[stability] layer=3 role={role}", flush=True)
        records.extend(_run_record(parent, raw, role, device))
    rule_report = _rule_report(records)
    elapsed = time.perf_counter() - started

    output = {
        "protocol": "proxy-v2",
        "diagnostic": "l3-fc-cross-fold-stability-v1",
        "scope": "research-oracle",
        "status": "ok",
        "decision": "no_stable_fixed_rule_observed",
        "parent": str(parent_path),
        "parent_sha256": parent_hash,
        "cache": str(args.cache.resolve()),
        "cache_metadata": raw.metadata,
        "device": str(device),
        "layers": list(LAYERS),
        "roles": list(ROLES),
        "folds": list(FOLDS),
        "teacher_mode": "joint bounded legal teacher with one batched Jacobi mantissa pass",
        "wall_seconds": _finite(elapsed),
        "records": records,
        "stability": rule_report,
        "note": "Diagnostic only; no threshold/LUT is written to an API state and no v155 is created.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")

    lines = [
        "# L3 fc cross-fold stability probe — proxy-v2",
        "",
        f"- parent SHA256: `{parent_hash}`",
        f"- scope: layer `{LAYERS[0]}`, roles `{', '.join(ROLES)}`, folds `{FOLDS}`",
        f"- teacher: bounded joint legal teacher + one batched Jacobi pass",
        f"- wall: `{elapsed:.3f}s` (research diagnostic; not candidate API time)",
        "- decision: `no_stable_fixed_rule_observed` (diagnostic label; no promotion gate)",
        "",
        "## Fold-paired teacher decisions",
        "",
        "| role | fold0 joint margin | fold1 joint margin | exact block sign agreement | exact margin correlation |",
        "|---|---:|---:|---:|---:|",
    ]
    for role in ROLES:
        role_records = [item for item in records if item["role"] == role]
        by_fold = {int(item["fold"]): item for item in role_records}
        left = [float(item["exact_single_block_margin"]) for item in by_fold[0]["blocks"]]
        right = [float(item["exact_single_block_margin"]) for item in by_fold[1]["blocks"]]
        role_rules = rule_report["by_role"][role]
        lines.append(
            f"| {role} | {by_fold[0]['teacher_joint_margin']:.6f} | {by_fold[1]['teacher_joint_margin']:.6f} | "
            f"{role_rules['decision_sign_agreement']}/{role_rules['block_count']} | "
            f"{role_rules['exact_margin_correlation'] if role_rules['exact_margin_correlation'] is not None else 'NA'} |"
        )
    lines.extend([
        "",
        "## Feature correlations",
        "",
        "| role | feature | Pearson fold0↔fold1 | Spearman fold0↔fold1 | best fold0 rule held-out accuracy |",
        "|---|---|---:|---:|---:|",
    ])
    for role in ROLES:
        role_result = rule_report["by_role"][role]
        best = role_result.get("best_train_rule")
        for feature, value in role_result["features"].items():
            heldout = "NA"
            if best is not None and best["feature"] == feature:
                heldout = f"{best['heldout']['accuracy']:.3f}"
            lines.append(
                f"| {role} | {feature} | {value['pearson'] if value['pearson'] is not None else 'NA'} | "
                f"{value['spearman'] if value['spearman'] is not None else 'NA'} | {heldout} |"
            )
    lines.extend([
        "",
        "## Block-level records",
        "",
        "| role | fold | block | exact single margin | quadratic margin | target amax | error rms |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ])
    for record in records:
        for block in record["blocks"]:
            features = block["features"]
            lines.append(
                f"| {record['role']} | {record['fold']} | {block['block']} | "
                f"{block['exact_single_block_margin']:.6f} | {block['quadratic_margin']:.6f} | "
                f"{features['target_amax']:.5f} | {features['parent_error_rms']:.5f} |"
            )
    lines.extend([
        "",
        "The threshold/LUT rows are descriptive only.  A rule fitted on one fold must be judged on the",
        "other fold; the probe does not compile or store any teacher decision.  If feature rank/sign is",
        "not stable, the L3 fc representation family remains closed and the next implementation is L2",
        "with a deployed-Q(W) output metric rather than another s_q/s_d or unconstrained pair balance.",
    ])
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
