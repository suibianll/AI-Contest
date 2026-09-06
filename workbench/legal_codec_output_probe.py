"""R0/R1 evidence probe for the legal HiF4 codec and output objective.

This workbench tool deliberately stays outside ``solution.py``.  R0 checks the
old codebook evidence chain with small, deterministic tests.  R1 computes the
fixed-coordinate, legal five-field block optimum from the evaluator-owned
codec and compares it with the v186 parent on a fixed sample of the proxy-v2
cache.  Later stages are intentionally not inferred from a failed R0/R1 run.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import math
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch


ROOT = Path(__file__).resolve().parents[1]
for _path in (ROOT / "evaluator", ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import official_eval as v2  # noqa: E402
import proxy_v3_eval as v3  # noqa: E402
import reference_hif4 as ref  # noqa: E402
import solution as sol  # noqa: E402


E4M3_MIN_SUBNORMAL = 2.0 ** -9
E4M3_MIN_NORMAL = 2.0 ** -6
E4M3_MAX = 448.0
E2M1_VALUES = (0.0, -0.5, 0.5, -1.0, 1.0, -1.5, 1.5, -2.0, 2.0,
               -3.0, 3.0, -4.0, 4.0, -6.0, 6.0)
E6M2_CODES = tuple(range(255))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _git_head() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True,
            capture_output=True, text=True,
        )
        return result.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _torch_environment(device: str) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "torch": str(torch.__version__),
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_version": str(torch.version.cuda),
        "requested_device": str(device),
    }
    if torch.cuda.is_available():
        index = torch.cuda.current_device()
        payload.update({
            "cuda_device_index": int(index),
            "gpu": torch.cuda.get_device_name(index),
            "gpu_capability": list(torch.cuda.get_device_capability(index)),
        })
    else:
        payload["gpu"] = None
    return payload


def _uniform_indices(total: int, count: int) -> tuple[int, ...]:
    if total <= 0 or count <= 0:
        return ()
    if total == 1 or count == 1:
        return (0,)
    return tuple(sorted({int(math.floor(k * (total - 1) / (count - 1))) for k in range(count)}))


def finite_e4m3_scales() -> tuple[float, ...]:
    """Return every positive finite E4M3FN value used by the NVFP4 scale."""

    values = [n * E4M3_MIN_SUBNORMAL for n in range(1, 8)]
    for exponent_field in range(1, 15):
        exponent = exponent_field - 7
        values.extend(
            (2.0 ** exponent) * (1.0 + mantissa / 8.0)
            for mantissa in range(8)
        )
    # E4M3FN reserves the final mantissa code for NaN at exponent field 15.
    values.extend(2.0 ** 8 * (1.0 + mantissa / 8.0) for mantissa in range(7))
    values = sorted(set(values))
    if len(values) != 126 or values[0] != E4M3_MIN_SUBNORMAL or values[-1] != E4M3_MAX:
        raise AssertionError("unexpected E4M3 finite-value enumeration")
    return tuple(values)


def carrier_code_counts(carrier: torch.Tensor) -> dict[str, int]:
    """Count signed E2M1 carriers through their absolute code values."""

    values = carrier.detach().to(torch.float32).reshape(-1)
    counts: dict[str, int] = {}
    for code in (0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0):
        counts[str(code)] = int((values.abs() == code).sum())
    counts["zero"] = int((values == 0).sum())
    counts["total"] = int(values.numel())
    return counts


def compatibility_exact_mask(
    carrier: torch.Tensor,
    input_scale: torch.Tensor,
    sf: torch.Tensor,
    mantissa: torch.Tensor,
    lv2: torch.Tensor,
    lv3: torch.Tensor,
) -> torch.Tensor:
    """Check exact legal reconstruction against the *scaled* NVFP4 value.

    This is intentionally numeric rather than a table lookup.  It makes the
    two historical failure modes explicit: signs are retained and the dense
    target is ``carrier * input_scale``.  The returned mask is elementwise.
    """

    target = carrier.to(torch.float32) * input_scale.to(torch.float32)
    rebuilt = (
        torch.sign(carrier).to(torch.float32)
        * mantissa.to(torch.float32)
        * sf.to(torch.float32)
        * lv2.to(torch.float32)
        * lv3.to(torch.float32)
    )
    return target == rebuilt


def _codebook_bf16_probe() -> dict[str, Any]:
    scales = torch.tensor(finite_e4m3_scales(), dtype=torch.float32)
    carriers = torch.tensor(E2M1_VALUES, dtype=torch.float32)
    fp32 = carriers[:, None] * scales[None, :]
    bf16_roundtrip = fp32.to(torch.bfloat16).to(torch.float32)
    different = fp32 != bf16_roundtrip
    abs_error = (bf16_roundtrip - fp32).abs()
    rel_error = abs_error / fp32.abs().clamp_min(torch.finfo(torch.float32).tiny)
    flat_index = int(abs_error.reshape(-1).argmax())
    carrier_index, scale_index = divmod(flat_index, int(scales.numel()))

    # This second arm is not a codebook product.  It is a deterministic
    # transformed dense tensor, kept separate so its BF16 rounding cannot be
    # misreported as a raw E2M1/E4M3 compatibility failure.
    transformed_seed = torch.arange(128, dtype=torch.float32).reshape(2, 64)
    raw_dense = (0.12345 + transformed_seed * 0.00317).contiguous()
    transformed_dense = sol._block_hadamard_transform(raw_dense, 8, 0).to(torch.float32)
    transformed_bf16 = transformed_dense.to(torch.bfloat16).to(torch.float32)
    transformed_diff = transformed_dense != transformed_bf16

    return {
        "finite_e4m3_scale_count": int(scales.numel()),
        "e2m1_signed_carrier_count": int(carriers.numel()),
        "raw_codebook_product_count": int(fp32.numel()),
        "raw_fp32_vs_bf16_bitwise_differences": int(different.sum()),
        "raw_max_abs_error": float(abs_error.max()),
        "raw_max_relative_error": float(rel_error.max()),
        "raw_max_error_example": {
            "carrier": float(carriers[carrier_index]),
            "scale": float(scales[scale_index]),
            "fp32": float(fp32.reshape(-1)[flat_index]),
            "bf16_roundtrip": float(bf16_roundtrip.reshape(-1)[flat_index]),
        },
        "transformed_dense_element_count": int(transformed_dense.numel()),
        "transformed_dense_fp32_vs_bf16_bitwise_differences": int(transformed_diff.sum()),
        "transformed_dense_max_abs_error": float((transformed_bf16 - transformed_dense).abs().max()),
        "transformed_dense_is_separate_from_raw_codebook": True,
    }


def _regression_checks() -> dict[str, Any]:
    checks: dict[str, Any] = {}

    carrier = torch.tensor([[-1.0, 1.0, -0.5, 0.5, 0.0, -6.0, 6.0]])
    counts = carrier_code_counts(carrier)
    checks["signed_carrier_count"] = {
        "pass": counts["1.0"] == 2 and counts["0.5"] == 2 and counts["6.0"] == 2,
        "counts": counts,
    }

    # E6M2 code 184 decodes to 0.25.  The negative carrier is exact only when
    # the NVFP4 input scale is applied; omitting it reconstructs -1.0.
    scaled = compatibility_exact_mask(
        torch.tensor([-1.0]), torch.tensor([0.25]),
        torch.tensor([0.25]), torch.tensor([1.0]),
        torch.tensor([1.0]), torch.tensor([1.0]),
    )
    unscaled = compatibility_exact_mask(
        torch.tensor([-1.0]), torch.tensor([1.0]),
        torch.tensor([0.25]), torch.tensor([1.0]),
        torch.tensor([1.0]), torch.tensor([1.0]),
    )
    checks["compatibility_uses_input_scale"] = {
        "pass": bool(scaled.item()) and not bool(unscaled.item()),
        "scaled_exact": bool(scaled.item()),
        "unscaled_exact": bool(unscaled.item()),
    }

    small_code, _ = ref.standard_e6m2_scale(torch.tensor([0.125]))
    large_code, _ = ref.standard_e6m2_scale(torch.tensor([32.0]))
    checks["sf_seed_is_data_dependent"] = {
        "pass": int(small_code.item()) != int(large_code.item()),
        "small_amax_code": int(small_code.item()),
        "large_amax_code": int(large_code.item()),
    }

    torch.manual_seed(20260905)
    legal = ref.encode_standard_hif4(torch.randn(2, 64))
    ref.validate_hif4_params(legal, (2, 64))
    invalid_shape = dict(legal)
    invalid_shape["scale_lv3"] = legal["scale_lv3"].repeat(1, 1, 2, 1, 1)
    shape_rejected = False
    try:
        ref.validate_hif4_params(invalid_shape, (2, 64))
    except ValueError:
        shape_rejected = True
    checks["official_hierarchy_shapes"] = {
        "pass": shape_rejected,
        "legal_shapes": {name: list(value.shape) for name, value in legal.items()},
        "rejects_non_shared_lv3": shape_rejected,
    }

    # A product target is intentionally different from the operand target.
    x = torch.tensor([[1.0, 0.5, -0.75, 2.0] * 16, [-1.25, 0.25, 0.5, -2.5] * 16])
    w = torch.tensor([[0.5, -1.0, 2.0, 0.25] * 16, [1.5, 0.75, -0.5, 2.5] * 16])
    xq = ref.dequantize_hif4(ref.encode_standard_hif4(x), x.shape)
    wq = ref.dequantize_hif4(ref.encode_standard_hif4(w), w.shape)
    operand_mse = float((xq - x).square().mean() + (wq - w).square().mean())
    output_mse = float((xq @ wq.T - x @ w.T).square().mean())
    checks["operand_vs_output_objective"] = {
        "pass": not math.isclose(operand_mse, output_mse, rel_tol=1e-8, abs_tol=1e-12),
        "operand_mse": operand_mse,
        "output_mse": output_mse,
    }

    checks["all_pass"] = all(bool(item.get("pass")) for item in checks.values())
    return checks


def _novelty_table(parent_source: str) -> list[dict[str, Any]]:
    source = Path(parent_source).read_text(encoding="utf-8")
    return [
        {
            "target": "R1 legal hierarchy exact block",
            "variables": "all 255 E6M2 sf; shared lv2/lv3; 64 mantissa values",
            "joint_update": "one 64-element block, exact given sf",
            "online_cost": "diagnostic only",
            "parent": "v186",
            "prior_call_sites": 0,
        },
        {
            "target": "_jdrq_refine_mantissa_coordinates",
            "variables": "selected mantissa coordinates",
            "joint_update": "coordinate passes on selected blocks",
            "online_cost": "zero after calibration state compilation",
            "parent": "v186",
            "prior_call_sites": source.count("_jdrq_refine_mantissa_coordinates("),
        },
        {
            "target": "L3/full64 and cross-fold minimax",
            "variables": "64-block/group code updates and fold aggregation",
            "joint_update": "full64 or cross-fold candidate blocks",
            "online_cost": "state-dependent; historical research",
            "parent": "v186/v166",
            "prior_call_sites": source.count("_refine_weight_blocks64(") + source.count("_refine_weight_blocks_cross64("),
        },
        {
            "target": "R2-L output joint rounding",
            "variables": "fixed 8 mantissa/sign codes per group",
            "joint_update": "256 legal combinations, one pass, actual X_hat output",
            "online_cost": "zero after calibration; candidate is not yet implemented",
            "parent": "v186 validation, v180 deployment",
            "prior_call_sites": 0,
        },
        {
            "target": "R2-A attention output oracle",
            "variables": "fixed R1 legal block substitutions",
            "joint_update": "Q-only/K-only offline substitutions",
            "online_cost": "oracle only unless a static rule is found",
            "parent": "v186",
            "prior_call_sites": 0,
        },
        {
            "target": "CAT/Trellis/Babai historical arms",
            "variables": "group transforms or finite code paths",
            "joint_update": "already tested historical families",
            "online_cost": "historical evidence only",
            "parent": "v166/v168",
            "prior_call_sites": 0,
        },
    ]


def _manifest(
    stage: str,
    args: argparse.Namespace,
    config_path: Path,
    cache_path: Path,
    parent_path: Path,
    status: str,
    expected_cases: Mapping[str, Any],
    executed_cases: Mapping[str, Any],
) -> dict[str, Any]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    manifest = {
        "stage": stage,
        "status": status,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "git_head": _git_head(),
        "parent": str(parent_path.resolve()),
        "parent_sha256": sha256_file(parent_path),
        "tool": str(Path(__file__).resolve()),
        "tool_sha256": sha256_file(Path(__file__).resolve()),
        "config": str(config_path.resolve()),
        "config_sha256": sha256_file(config_path),
        "reference_hif4": str((ROOT / "evaluator" / "reference_hif4.py").resolve()),
        "reference_hif4_sha256": sha256_file(ROOT / "evaluator" / "reference_hif4.py"),
        "cache": str(cache_path.resolve()),
        "cache_sha256": sha256_file(cache_path),
        "environment": _torch_environment(args.device),
        "command": [str(value) for value in sys.argv],
        "config_snapshot": config,
        "expected_cases": dict(expected_cases),
        "executed_cases": dict(executed_cases),
        "dtype": "FP64 solver / FP32 reference tensors",
    }
    return manifest


def _write_stage(output_dir: Path, result: Mapping[str, Any], manifest: Mapping[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    (output_dir / "run_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    lines = [
        f"# legal codec/output probe: {result['stage']}", "",
        f"- status: `{result['status']}`",
        f"- hypothesis: {result['hypothesis']}",
        f"- parent SHA256: `{manifest['parent_sha256']}`",
        f"- cache SHA256: `{manifest['cache_sha256']}`",
        f"- expected cases: `{json.dumps(result['expected_cases'], ensure_ascii=False, sort_keys=True)}`",
        f"- executed cases: `{json.dumps(result['executed_cases'], ensure_ascii=False, sort_keys=True)}`",
        "",
        f"## Gate: {result['gate']}", "",
        f"{result['reason']}", "",
    ]
    if result["stage"] == "r0":
        lines.extend([
            "## R0 regression checks", "",
            f"```json\n{json.dumps(result['metrics']['regression_checks'], ensure_ascii=False, indent=2)}\n```",
            "", "## BF16 probe", "",
            f"```json\n{json.dumps(result['metrics']['bf16_probe'], ensure_ascii=False, indent=2)}\n```",
            "", "## Novelty table", "",
            "| target | variables | joint update | online cost | parent |", "|---|---|---|---|---|",
        ])
        for item in result["metrics"]["novelty_table"]:
            lines.append(
                f"| {item['target']} | {item['variables']} | {item['joint_update']} | "
                f"{item['online_cost']} | {item['parent']} |"
            )
    elif result["stage"] == "r1":
        gate = result["metrics"].get("g1_a", {})
        lines.extend([
            "## G1-A", "",
            f"- median improvement: `{gate.get('median_improvement', float('nan')):.6f}`",
            f"- positive layers: `{gate.get('positive_layers', 0)}/{gate.get('layer_count', 0)}`",
            f"- legal witness: `{gate.get('legal_witness', False)}`",
            "", "## Layer summaries", "",
            "| layer | median improvement | positive block fraction | exact violations |", "|---:|---:|---:|---:|",
        ])
        for item in result["metrics"].get("layer_summary", []):
            lines.append(
                f"| {item['layer']} | {item['median_improvement']:+.6f} | "
                f"{item['positive_fraction']:.3f} | {item['exact_violations']} |"
            )
    elif result["stage"] == "r2-linear":
        gate = result["metrics"].get("g2_l", {})
        lines.extend([
            "## G2-L", "",
            f"- LOO gate passes: `{gate.get('loo_gate_passes', 0)}/{gate.get('focus_state_count', 0)}`",
            f"- LOO gate layers: `{gate.get('loo_gate_layers', [])}`",
            f"- holdout mean/median gain: `{gate.get('holdout_mean_gain', 0.0):+.6f}` / `"
            f"`{gate.get('holdout_median_gain', 0.0):+.6f}`",
            f"- holdout split means: `{json.dumps(gate.get('holdout_split_mean_gain', {}), ensure_ascii=False, sort_keys=True)}`",
            f"- holdout L1 mean absolute case gain: `{gate.get('holdout_l1_mean_abs_case_gain', 0.0):.6f}`",
            f"- barrier witnesses: `{gate.get('barrier_witness_count', 0)}`",
            "", "## State summary", "",
            "| layer | role | LOO median | LOO worst | holdout mean | status |", "|---:|---|---:|---:|---:|---|",
        ])
        for item in result["metrics"].get("records", []):
            if "loo" not in item:
                lines.append(
                    f"| {item.get('layer', '?')} | {item.get('role', '?')} | — | — | — | {item.get('status', '?')} |"
                )
                continue
            lines.append(
                f"| {item['layer']} | {item['role']} | {item['loo']['median_gain']:+.6f} | "
                f"{item['loo']['worst_gain']:+.6f} | {item['holdout']['mean_gain']:+.6f} | {item['status']} |"
            )
    elif result["stage"] == "r2-attention":
        gate = result["metrics"].get("g2_a", {})
        lines.extend([
            "## G2-A", "",
            f"- Q oracle mean gain: `{gate.get('q_oracle_mean_gain', 0.0):+.6f}`",
            f"- K oracle mean gain: `{gate.get('k_oracle_mean_gain', 0.0):+.6f}`",
            f"- deployable rule: `{gate.get('deployable_rule', False)}`",
            "- This is an offline block-replacement oracle; no runtime joint Q/K/V rule is registered.",
        ])
    (output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_r0(args: argparse.Namespace, config: Mapping[str, Any], config_path: Path,
           output_dir: Path, parent_path: Path, cache_path: Path) -> dict[str, Any]:
    started = time.perf_counter()
    checks = _regression_checks()
    bf16 = _codebook_bf16_probe()
    raw = v2.load_pack(cache_path)
    cache_metadata = {
        "protocol": raw.metadata.get("protocol"),
        "model": raw.metadata.get("model"),
        "layers": int(raw.layers),
        "hidden_size": int(raw.hidden_size),
        "q_heads": int(raw.q_heads),
        "kv_heads": int(raw.kv_heads),
        "head_dim": int(raw.head_dim),
        "calibration_lengths": [len(item.input_ids) for item in raw.calibration_windows],
        "test_lengths": [len(item.input_ids) for item in raw.test_windows],
    }
    expected = {
        "finite_e4m3_scales": 126,
        "signed_e2m1_carrier_values": 15,
        "raw_codebook_products": 1890,
        "synthetic_regression_groups": 5,
        "cache_protocol": "proxy-v2",
    }
    executed = {
        "finite_e4m3_scales": bf16["finite_e4m3_scale_count"],
        "signed_e2m1_carrier_values": bf16["e2m1_signed_carrier_count"],
        "raw_codebook_products": bf16["raw_codebook_product_count"],
        "synthetic_regression_groups": len(checks) - 1,
        "cache_loaded": True,
    }
    status = "PASS" if checks["all_pass"] and cache_metadata["protocol"] == "proxy-v2" else "ERROR"
    result = {
        "stage": "r0",
        "status": status,
        "hypothesis": "旧 cb1/cb2 证据链存在实现缺陷；修复后须先证明合法字段、输入 scale、符号和输出目标路径均正确。",
        "expected_cases": expected,
        "executed_cases": executed,
        "failed_cases": [] if status == "PASS" else [name for name, item in checks.items() if isinstance(item, dict) and not item.get("pass", True)],
        "metrics": {
            "cache_metadata": cache_metadata,
            "regression_checks": checks,
            "bf16_probe": bf16,
            "novelty_table": _novelty_table(str(parent_path)),
            "elapsed_s": time.perf_counter() - started,
        },
        "gate": "G0_PASS" if status == "PASS" else "G0_ERROR",
        "reason": (
            "所有固定 R0 case、官方解码合法性检查和 BF16 分离测试通过；可进入 R1。"
            if status == "PASS" else
            "R0 有未通过的固定回归或 cache 协议错误；不得进入后续阶段。"
        ),
        "next_step": "r1" if status == "PASS" else "repair_r0",
    }
    manifest = _manifest("r0", args, config_path, cache_path, parent_path, status, expected, executed)
    _write_stage(output_dir, result, manifest)
    return result


def _state_parent_transform(weight: torch.Tensor, state: Mapping[str, Any]) -> torch.Tensor:
    device = weight.device
    channels = int(weight.shape[-1])
    smooth_inv = state.get("smooth_inv")
    if smooth_inv is None:
        d = torch.ones(channels, dtype=torch.float32, device=device)
    else:
        d = smooth_inv.to(device=device, dtype=torch.float32).reshape(-1).reciprocal()
    permutation = state.get("permutation")
    if permutation is None:
        order = torch.arange(channels, device=device, dtype=torch.int64)
    else:
        order = permutation.to(device=device, dtype=torch.int64).reshape(-1)
    transformed = sol._linear_pair_transform(
        weight.to(torch.float32), d, order,
        int(state.get("block_smooth_size", 0)),
        int(state.get("block_smooth_seed", 0)),
        weight_side=True,
        cat_transform=state.get("cat_transform"),
    )
    residual_u = state.get("residual_u")
    residual_v = state.get("residual_v")
    if residual_u is None or residual_v is None:
        rank1_u = state.get("rank1_u")
        rank1_v = state.get("rank1_v")
        if rank1_u is not None and rank1_v is not None and float(rank1_u.square().sum()) > 0.0:
            residual_u = rank1_u.reshape(channels, 1)
            residual_v = rank1_v.reshape(channels, 1)
    if residual_u is not None and residual_v is not None:
        u = residual_u.to(device=device, dtype=torch.float32)
        v = residual_v.to(device=device, dtype=torch.float32)
        transformed = transformed - (transformed @ v) @ u.transpose(0, 1)
    return transformed


def _state_parent_inverse(weight: torch.Tensor, state: Mapping[str, Any]) -> torch.Tensor:
    device = weight.device
    channels = int(weight.shape[-1])
    residual_u = state.get("residual_u")
    residual_v = state.get("residual_v")
    if residual_u is None or residual_v is None:
        rank1_u = state.get("rank1_u")
        rank1_v = state.get("rank1_v")
        if rank1_u is not None and rank1_v is not None and float(rank1_u.square().sum()) > 0.0:
            residual_u = rank1_u.reshape(channels, 1)
            residual_v = rank1_v.reshape(channels, 1)
    restored = weight.to(torch.float32)
    if residual_u is not None and residual_v is not None:
        u = residual_u.to(device=device, dtype=torch.float32)
        v = residual_v.to(device=device, dtype=torch.float32)
        restored = restored + (restored @ v) @ u.transpose(0, 1)
    cat_transform = state.get("cat_transform")
    if cat_transform is not None:
        restored = sol._apply_cat64_rows(restored, cat_transform, inverse=False)
    block_size = int(state.get("block_smooth_size", 0))
    if block_size:
        restored = sol._block_hadamard_transform(
            restored, block_size, int(state.get("block_smooth_seed", 0))
        )
    permutation = state.get("permutation")
    if permutation is None:
        order = torch.arange(channels, device=device, dtype=torch.int64)
    else:
        order = permutation.to(device=device, dtype=torch.int64).reshape(-1)
    inverse_order = torch.argsort(order)
    restored = restored.index_select(-1, inverse_order)
    smooth_inv = state.get("smooth_inv")
    if smooth_inv is None:
        d = torch.ones(channels, dtype=torch.float32, device=device)
    else:
        d = smooth_inv.to(device=device, dtype=torch.float32).reshape(-1).reciprocal()
    return restored * d.reciprocal().reshape(1, -1)


def _sample_blocks(matrix: torch.Tensor, rows: Sequence[int], blocks: Sequence[int]) -> torch.Tensor:
    parts = [matrix[int(row), int(block) * 64:(int(block) + 1) * 64] for row in rows for block in blocks]
    return torch.stack(parts, dim=0).to(torch.float32)


def exact_legal_blocks(values: torch.Tensor) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    """Solve the legal 64-block operand SSE exactly with FP64 arithmetic."""

    x = values.detach().to(torch.float64).reshape(-1, 64)
    grouped = x.abs().reshape(-1, 8, 2, 4)
    signs = torch.sign(x).reshape(-1, 8, 2, 4).to(torch.float32)
    count = int(grouped.shape[0])
    best_loss = torch.full((count,), float("inf"), dtype=torch.float64, device=x.device)
    best_code = torch.zeros(count, dtype=torch.int64, device=x.device)
    best_lv2 = torch.ones((count, 8), dtype=torch.float32, device=x.device)
    best_lv3 = torch.ones((count, 8, 2), dtype=torch.float32, device=x.device)
    best_mant = torch.zeros((count, 8, 2, 4), dtype=torch.float32, device=x.device)
    codes = torch.arange(255, dtype=torch.int64, device=x.device)
    scales = ref.e6m2_decode(codes).to(torch.float64)

    for code, scale in zip(codes.tolist(), scales):
        losses: list[torch.Tensor] = []
        mantissas: list[torch.Tensor] = []
        for exponent in (0, 1, 2):
            denominator = scale * float(1 << exponent)
            mantissa = (grouped * (4.0 / denominator)).round().clamp(0.0, 7.0) * 0.25
            losses.append((grouped - mantissa * denominator).square().sum(dim=-1))
            mantissas.append(mantissa.to(torch.float32))
        loss0, loss1, loss2 = losses
        choose_12 = loss2 < loss1
        cost_01 = torch.minimum(loss0, loss1).sum(dim=-1)
        cost_12 = torch.minimum(loss1, loss2).sum(dim=-1)
        use_12 = cost_12 < cost_01
        choose_23 = torch.where(use_12[..., None], choose_12, loss1 < loss0)
        selected_loss = torch.where(use_12, cost_12, cost_01).sum(dim=-1)
        better = selected_loss < best_loss
        if not bool(better.any()):
            continue
        total_exponent = use_12[..., None].to(torch.int64) + choose_23.to(torch.int64)
        stack = torch.stack(mantissas, dim=3)  # [N,8,2,3,4]
        gather_index = total_exponent.unsqueeze(-1).unsqueeze(-1).expand(-1, 8, 2, 1, 4)
        selected_mant = torch.gather(stack, 3, gather_index).squeeze(3)
        best_loss = torch.where(better, selected_loss, best_loss)
        best_code = torch.where(better, torch.tensor(code, device=x.device), best_code)
        best_lv2 = torch.where(better[:, None], 1.0 + use_12.to(torch.float32), best_lv2)
        best_lv3 = torch.where(better[:, None, None], 1.0 + choose_23.to(torch.float32), best_lv3)
        best_mant = torch.where(better[:, None, None, None], selected_mant, best_mant)

    params = {
        "scale_factor": ref.e6m2_decode(best_code).to(torch.float32).reshape(count, 1, 1, 1, 1),
        "scale_lv2": best_lv2.reshape(count, 1, 8, 1, 1),
        "scale_lv3": best_lv3.reshape(count, 1, 8, 2, 1),
        "sign": torch.where(best_mant == 0.0, torch.zeros_like(signs), signs).reshape(count, 1, 8, 2, 4),
        "mant": best_mant.reshape(count, 1, 8, 2, 4),
    }
    ref.validate_hif4_params(params, (count, 64))
    return params, best_loss


def _load_parent_state(
    raw: Any, parent_path: Path, layer: int, role: str, device: str,
    prepared_cache: dict[int, tuple[Any, dict[tuple[int, str], tuple[Any, dict[str, torch.Tensor]]]]],
) -> tuple[Any, dict[str, torch.Tensor], str]:
    shard = int(layer) % int(v3.SHARD_COUNT)
    if shard not in prepared_cache:
        pack = v3.prepare_shard(raw, shard, "both", ood=False)
        identity = v3._calibration_identity(parent_path.resolve(), pack, torch.device(device))
        artifact = v3.default_calibration_cache_path(identity)
        if not artifact.is_file():
            raise FileNotFoundError(f"verified v186 calibration artifact is missing: {artifact}")
        weights, _ = v3.load_calibration_artifact(artifact, identity, pack)
        prepared_cache[shard] = (artifact, weights)
    artifact, weights = prepared_cache[shard]
    state, params = weights[(int(layer), str(role))]
    return state, params, str(artifact.resolve())


def _arm_metrics(reference_dense: torch.Tensor, decoded: torch.Tensor) -> dict[str, float]:
    error = (decoded.to(torch.float32) - reference_dense.to(torch.float32)).square()
    return {"sse": float(error.sum()), "mse": float(error.mean())}


def _parent_encoder_only_params(dense: torch.Tensor) -> dict[str, torch.Tensor]:
    """Run the v186 encoder-only arm with the fixed parent weight settings."""

    refine_ratio = (
        sol._WEIGHT_REFINE_MAX_RATIO_SMALL
        if int(dense.numel()) <= 4_194_304
        else sol._WEIGHT_REFINE_MAX_RATIO_LARGE
    )
    return sol._dense_to_hif4(
        dense,
        search_offsets=sol._WEIGHT_OFFSETS,
        error_threshold=sol._WEIGHT_REFINE_ERROR_THRESHOLD,
        accept_margin=sol._WEIGHT_REFINE_ACCEPT_MARGIN,
        max_refine_ratio=refine_ratio,
        max_refine_blocks=sol._WEIGHT_REFINE_MAX_BLOCKS,
        full_sweep_top_k=sol._WEIGHT_FULL_SWEEP_TOP_K,
    )


def run_r1(args: argparse.Namespace, config: Mapping[str, Any], config_path: Path,
           output_dir: Path, parent_path: Path, cache_path: Path) -> dict[str, Any]:
    started = time.perf_counter()
    raw = v2.load_pack(cache_path)
    layers = [int(v) for v in config["r1"]["layers"]]
    roles = [str(v) for v in config["r1"]["roles"]]
    sample_rows = int(config["r1"]["rows"])
    sample_blocks = int(config["r1"]["blocks"])
    prepared_cache: dict[int, tuple[Any, dict[tuple[int, str], tuple[Any, dict[str, torch.Tensor]]]]] = {}
    records: list[dict[str, Any]] = []
    witness: dict[str, Any] | None = None
    for layer in layers:
        if layer < 0 or layer >= raw.layers:
            raise ValueError(f"R1 layer {layer} is outside cache")
        for role in roles:
            if role not in raw.weights[layer]:
                raise ValueError(f"R1 role {role} is absent at layer {layer}")
            weight_pair = v2._pair(raw.weights[layer][role])
            raw_dense = v2.dequantize_nvfp4(*weight_pair).to(torch.float32)
            state, parent_params, artifact = _load_parent_state(
                raw, parent_path, layer, role, args.device, prepared_cache
            )
            parent_coord = _state_parent_transform(raw_dense, state)
            parent_deployed_coord = ref.dequantize_hif4(parent_params, tuple(raw_dense.shape)).to(torch.float32)
            parent_deployed_raw = _state_parent_inverse(parent_deployed_coord, state)
            encoder_coord_params = _parent_encoder_only_params(parent_coord)
            encoder_coord = sol._dequantize_hif4(encoder_coord_params).to(torch.float32)
            encoder_raw_params = _parent_encoder_only_params(raw_dense)
            encoder_raw = sol._dequantize_hif4(encoder_raw_params).to(torch.float32)
            standard_coord = ref.decode_standard_hif4(ref.encode_standard_hif4(parent_coord)).to(torch.float32)
            standard_raw = ref.decode_standard_hif4(ref.encode_standard_hif4(raw_dense)).to(torch.float32)
            rows = _uniform_indices(int(raw_dense.shape[0]), sample_rows)
            blocks = _uniform_indices(int(raw_dense.shape[1]) // 64, sample_blocks)
            sampled_raw = _sample_blocks(raw_dense, rows, blocks)
            sampled_coord = _sample_blocks(parent_coord, rows, blocks)
            exact_raw_params, exact_raw_loss = exact_legal_blocks(sampled_raw)
            exact_coord_params, exact_coord_loss = exact_legal_blocks(sampled_coord)
            exact_raw = ref.dequantize_hif4(exact_raw_params, (int(sampled_raw.shape[0]), 64)).to(torch.float32)
            exact_coord = ref.dequantize_hif4(exact_coord_params, (int(sampled_coord.shape[0]), 64)).to(torch.float32)
            sampled_encoder_raw = _sample_blocks(encoder_raw, rows, blocks)
            sampled_encoder_coord = _sample_blocks(encoder_coord, rows, blocks)
            sampled_standard_raw = _sample_blocks(standard_raw, rows, blocks)
            sampled_standard_coord = _sample_blocks(standard_coord, rows, blocks)
            sampled_parent_raw = _sample_blocks(parent_deployed_raw, rows, blocks)
            sampled_parent_coord = _sample_blocks(parent_deployed_coord, rows, blocks)

            raw_parent_sse = (sampled_parent_raw - sampled_raw).square().sum(dim=1)
            coord_parent_sse = (sampled_parent_coord - sampled_coord).square().sum(dim=1)
            raw_encoder_sse = (sampled_encoder_raw - sampled_raw).square().sum(dim=1)
            coord_encoder_sse = (sampled_encoder_coord - sampled_coord).square().sum(dim=1)
            raw_exact_sse = (exact_raw - sampled_raw).square().sum(dim=1)
            coord_exact_sse = (exact_coord - sampled_coord).square().sum(dim=1)
            raw_standard_sse = (sampled_standard_raw - sampled_raw).square().sum(dim=1)
            coord_standard_sse = (sampled_standard_coord - sampled_coord).square().sum(dim=1)

            coord_improvement = (coord_encoder_sse - coord_exact_sse) / coord_encoder_sse.clamp_min(1e-30)
            raw_improvement = (raw_encoder_sse - raw_exact_sse) / raw_encoder_sse.clamp_min(1e-30)
            tolerance = 1e-10 + 1e-7 * coord_encoder_sse
            violations = int((coord_exact_sse > coord_encoder_sse + tolerance).sum())
            if witness is None and bool((coord_improvement > 1e-6).any()):
                index = int(torch.argmax(coord_improvement))
                witness = {
                    "layer": layer,
                    "role": role,
                    "row": int(rows[index // len(blocks)]),
                    "block": int(blocks[index % len(blocks)]),
                    "improvement": float(coord_improvement[index]),
                    "input_values": sampled_coord[index].tolist(),
                    "exact_params": {name: value[index].cpu().tolist() for name, value in exact_coord_params.items()},
                }
            records.append({
                "layer": layer,
                "role": role,
                "shape": list(raw_dense.shape),
                "rows": list(rows),
                "blocks": list(blocks),
                "parent_calibration_artifact": artifact,
                "raw": {
                    "standard": _arm_metrics(sampled_raw, sampled_standard_raw),
                    "encoder_only": _arm_metrics(sampled_raw, sampled_encoder_raw),
                    "deployed": _arm_metrics(sampled_raw, sampled_parent_raw),
                    "exact": _arm_metrics(sampled_raw, exact_raw),
                    "median_improvement_exact_vs_encoder": float(torch.median(raw_improvement)),
                },
                "final_transformed": {
                    "standard": _arm_metrics(sampled_coord, sampled_standard_coord),
                    "encoder_only": _arm_metrics(sampled_coord, sampled_encoder_coord),
                    "deployed": _arm_metrics(sampled_coord, sampled_parent_coord),
                    "exact": _arm_metrics(sampled_coord, exact_coord),
                    "median_improvement_exact_vs_encoder": float(torch.median(coord_improvement)),
                    "positive_blocks": int((coord_improvement > 0).sum()),
                    "block_count": int(coord_improvement.numel()),
                    "exact_violations": violations,
                },
            })
    focus_roles = {str(value) for value in config["r1"]["focus_roles"]}
    layer_summary = []
    for layer in layers:
        selected = [item for item in records if item["layer"] == layer and item["role"] in focus_roles]
        improvements = torch.tensor([
            item["final_transformed"]["median_improvement_exact_vs_encoder"] for item in selected
        ], dtype=torch.float64)
        layer_summary.append({
            "layer": layer,
            "median_improvement": float(torch.median(improvements)) if selected else 0.0,
            "positive_fraction": float((improvements > 0).to(torch.float64).mean()) if selected else 0.0,
            "exact_violations": int(sum(item["final_transformed"]["exact_violations"] for item in selected)),
        })
    focus_improvements = torch.tensor(
        [item["final_transformed"]["median_improvement_exact_vs_encoder"] for item in records if item["role"] in focus_roles],
        dtype=torch.float64,
    )
    positive_layers = sum(item["median_improvement"] > 0 for item in layer_summary)
    g1 = {
        "median_improvement": float(torch.median(focus_improvements)) if focus_improvements.numel() else 0.0,
        "positive_layers": int(positive_layers),
        "layer_count": len(layer_summary),
        "legal_witness": witness is not None,
        "exact_violations": int(sum(item["final_transformed"]["exact_violations"] for item in records)),
    }
    g1_pass = (
        g1["median_improvement"] >= 0.01
        and positive_layers >= 3
        and bool(g1["legal_witness"])
        and g1["exact_violations"] == 0
    )
    status = "PASS" if g1_pass else "PASS_TO_R2"
    expected = {
        "layers": len(layers),
        "roles_per_layer": len(roles),
        "sample_rows_per_matrix": sample_rows,
        "sample_blocks_per_row": sample_blocks,
        "focus_layers_for_g1": len(layers),
    }
    executed = {
        "records": len(records),
        "sampled_blocks": sum(len(item["rows"]) * len(item["blocks"]) for item in records),
        "exact_violations": g1["exact_violations"],
        "parent_artifacts": sorted({item["parent_calibration_artifact"] for item in records}),
    }
    result = {
        "stage": "r1",
        "status": status,
        "hypothesis": "在最终部署坐标中，固定 64-block 合法 HiF4 全局精确解可能揭示父编码器遗漏的 operand gap。",
        "expected_cases": expected,
        "executed_cases": executed,
        "failed_cases": [] if g1["exact_violations"] == 0 else ["exact_solver_worse_than_parent"],
        "metrics": {
            "g1_a": g1,
            "layer_summary": layer_summary,
            "records": records,
            "legal_witness": witness,
            "elapsed_s": time.perf_counter() - started,
        },
        "gate": "G1_A_PASS_TO_R3_A" if g1_pass else "G1_A_NO_MATERIAL_OPERAND_GAP_IN_PANEL",
        "reason": (
            "最终坐标 exact block 解达到预注册 G1-A 门槛，下一步是登记最小编码缺口并构造 R3-A。"
            if g1_pass else
            "最终坐标 exact block 解未达到 1%/3 层门槛；不能把局部合法解差异当作可部署收益，按计划转入 R2 输出目标诊断。"
        ),
        "next_step": "r3-a" if g1_pass else "r2",
    }
    manifest = _manifest("r1", args, config_path, cache_path, parent_path, status, expected, executed)
    _write_stage(output_dir, result, manifest)
    return result


def _state_activation_transform(dense: torch.Tensor, state: Mapping[str, Any]) -> torch.Tensor:
    """Mirror the continuous prefix of v186's deployed Linear activation path."""

    transformed = dense.to(torch.float32)
    multiplier = state.get("smooth_inv")
    if multiplier is not None:
        transformed = transformed * multiplier.to(
            device=transformed.device, dtype=torch.float32
        ).reshape(1, -1)
    permutation = state.get("permutation")
    if permutation is not None:
        transformed = transformed.index_select(
            -1, permutation.to(device=transformed.device, dtype=torch.int64).reshape(-1)
        )
    block_size = int(state.get("block_smooth_size", 0))
    if block_size:
        transformed = sol._block_hadamard_transform(
            transformed, block_size, int(state.get("block_smooth_seed", 0))
        )
    residual_u = state.get("residual_u")
    residual_v = state.get("residual_v")
    if residual_u is not None and residual_v is not None:
        u = residual_u.to(device=transformed.device, dtype=torch.float32)
        v = residual_v.to(device=transformed.device, dtype=torch.float32)
        transformed = transformed + (transformed @ u) @ v.transpose(0, 1)
    else:
        rank1_u = state.get("rank1_u")
        rank1_v = state.get("rank1_v")
        if rank1_u is not None and rank1_v is not None:
            u = rank1_u.to(device=transformed.device, dtype=torch.float32)
            v = rank1_v.to(device=transformed.device, dtype=torch.float32)
            transformed = transformed + (transformed @ u).unsqueeze(-1) * v
    return transformed


def _weight_field_views(
    params: Mapping[str, torch.Tensor], shape: Sequence[int], device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    out_features, channels = map(int, shape)
    blocks = channels // 64
    scale = params["scale_factor"].to(device=device, dtype=torch.float32).reshape(
        out_features, blocks
    )
    lv2 = params["scale_lv2"].to(device=device, dtype=torch.float32).reshape(
        out_features, blocks, 8
    )
    lv3 = params["scale_lv3"].to(device=device, dtype=torch.float32).reshape(
        out_features, blocks, 8, 2
    )
    signed_codes = (
        params["sign"].to(device=device, dtype=torch.float32)
        * params["mant"].to(device=device, dtype=torch.float32)
        * 4.0
    ).reshape(out_features, blocks, 8, 2, 4).round().to(torch.int64)
    return scale, lv2, lv3, signed_codes


def _linear_nvfp4_fold(
    raw_value: torch.Tensor,
    state: Mapping[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    pair = v2._pair(raw_value)
    pair_device = v2._move_pair(pair, device)
    reference = v2.dequantize_nvfp4(*pair_device).to(torch.float32)
    params = sol.hif4_dynamic_quantize_activation(
        pair_device[0], pair_device[1], state
    )
    quantized = ref.dequantize_hif4(params, tuple(reference.shape)).to(torch.float32)
    transformed_reference = _state_activation_transform(reference, state)
    return {
        "reference": reference,
        "transformed_reference": transformed_reference,
        "quantized": quantized,
    }


def _linear_apply_deltas(
    fold: Mapping[str, Any], deltas: torch.Tensor,
    rows: Sequence[int], blocks: Sequence[int],
) -> torch.Tensor:
    residual = fold["residual"].clone()
    row_tensor = torch.as_tensor(rows, dtype=torch.int64, device=residual.device)
    for block_position, block in enumerate(blocks):
        block_delta = deltas[:, block_position, :]
        if not bool((block_delta.abs() > 0.0).any()):
            continue
        lo = int(block) * 64
        hi = lo + 64
        prediction = fold["quantized"][:, lo:hi].mm(block_delta.transpose(0, 1))
        residual[:, row_tensor] = residual[:, row_tensor] - prediction
    return residual


def _select_linear_joint_codes(
    folds: Sequence[Mapping[str, Any]], train_indices: Sequence[int],
    parent_weight: torch.Tensor, continuous_weight: torch.Tensor,
    fields: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    rows: Sequence[int], blocks: Sequence[int],
) -> dict[str, Any]:
    """Select one fixed 8-code group at a time against the real Z output."""

    if not train_indices:
        raise ValueError("R2-L requires at least one training fold")
    device = parent_weight.device
    scale, lv2, lv3, parent_codes = fields
    deltas = torch.zeros(
        (len(rows), len(blocks), 64), dtype=torch.float32, device=device
    )
    residuals = [fold["residual"].clone() for fold in folds]
    attempted = 0
    accepted = 0
    changed_codes = 0
    barrier_witness: dict[str, Any] | None = None
    group_records: list[dict[str, Any]] = []
    bit_values = torch.arange(256, dtype=torch.int64, device=device)
    bit_values = ((bit_values[:, None] >> torch.arange(8, device=device)) & 1).to(torch.bool)
    row_to_position = {int(row): index for index, row in enumerate(rows)}

    for block_position, block in enumerate(blocks):
        lo = int(block) * 64
        for row in rows:
            row_position = row_to_position[int(row)]
            for group in range(8):
                attempted += 1
                start = group * 8
                stop = start + 8
                denominator = (
                    scale[row, int(block)]
                    * lv2[row, int(block), group]
                    * lv3[row, int(block), group].repeat_interleave(4)
                    / 4.0
                )
                target_codes = continuous_weight[row, lo + start:lo + stop] / denominator
                floor_codes = torch.floor(target_codes).clamp(-7, 7).to(torch.int64)
                ceil_codes = torch.ceil(target_codes).clamp(-7, 7).to(torch.int64)
                candidates = torch.where(
                    bit_values, ceil_codes.unsqueeze(0), floor_codes.unsqueeze(0)
                )
                current = parent_codes[row, int(block), group].reshape(-1).clone()
                if not bool((candidates == current).all(dim=1).any()):
                    candidates = torch.cat((candidates, current.unsqueeze(0)), dim=0)
                delta = candidates.to(torch.float32) * denominator.unsqueeze(0)
                old = current.to(torch.float32) * denominator
                delta = delta - old.unsqueeze(0)

                train_scores = []
                for fold_index in train_indices:
                    z_group = folds[fold_index]["quantized"][:, lo + start:lo + stop]
                    residual_column = residuals[fold_index][:, int(row)]
                    predicted = z_group.mm(delta.transpose(0, 1))
                    train_scores.append(
                        (predicted.square() - 2.0 * predicted * residual_column[:, None]).mean(dim=0)
                    )
                scores = torch.stack(train_scores, dim=0).mean(dim=0)
                best_index = int(torch.argmin(scores))
                best_score = float(scores[best_index])
                single_mask = (candidates != current.unsqueeze(0)).sum(dim=1) == 1
                single_best = float(scores[single_mask].min()) if bool(single_mask.any()) else 0.0
                if (
                    barrier_witness is None
                    and bool(single_mask.any())
                    and single_best >= -1.0e-12
                    and best_score < -1.0e-12
                ):
                    barrier_witness = {
                        "row": int(row),
                        "block": int(block),
                        "group": int(group),
                        "single_best_delta": single_best,
                        "joint_delta": best_score,
                        "parent_codes": current.cpu().tolist(),
                        "selected_codes": candidates[best_index].cpu().tolist(),
                    }
                if best_score < -1.0e-12:
                    selected_delta = delta[best_index]
                    deltas[row_position, block_position, start:stop] = selected_delta
                    for fold_index in train_indices:
                        z_group = folds[fold_index]["quantized"][:, lo + start:lo + stop]
                        residuals[fold_index][:, int(row)] -= z_group.mv(selected_delta)
                    accepted += 1
                    changed_codes += int((candidates[best_index] != current).sum())
                    group_records.append({
                        "row": int(row), "block": int(block), "group": int(group),
                        "score_delta": best_score,
                        "changed_codes": int((candidates[best_index] != current).sum()),
                    })
    return {
        "deltas": deltas,
        "attempted_groups": attempted,
        "accepted_groups": accepted,
        "changed_codes": changed_codes,
        "barrier_witness": barrier_witness,
        "group_records": group_records,
    }


def _linear_fold_gain(
    fold: Mapping[str, Any], candidate_residual: torch.Tensor,
) -> tuple[float, float, float]:
    parent_mse = float(fold["residual"].square().mean())
    candidate_mse = float(candidate_residual.square().mean())
    gain = (parent_mse - candidate_mse) / max(parent_mse, 1.0e-30)
    return gain, parent_mse, candidate_mse


def _linear_holdout_fold(
    raw: Any, layer: int, role: str, window: int, state: Mapping[str, Any],
    continuous_weight: torch.Tensor, parent_weight: torch.Tensor,
    device: torch.device,
) -> dict[str, Any]:
    fold = _linear_nvfp4_fold(raw.test_activations[role][window][layer], state, device)
    teacher = fold["transformed_reference"].mm(continuous_weight.transpose(0, 1))
    fold["teacher"] = teacher
    fold["residual"] = teacher - fold["quantized"].mm(parent_weight.transpose(0, 1))
    fold["split"] = str(raw.test_windows[window].split)
    fold["length"] = int(len(raw.test_windows[window].input_ids))
    return fold


def run_r2_linear(
    args: argparse.Namespace, config: Mapping[str, Any], config_path: Path,
    output_dir: Path, parent_path: Path, cache_path: Path,
) -> dict[str, Any]:
    started = time.perf_counter()
    raw = v2.load_pack(cache_path)
    device = torch.device(args.device)
    settings = config["r2_linear"]
    layers = [int(value) for value in settings["focus_layers"]]
    roles = [str(value) for value in settings["focus_roles"]]
    calibration_indices = [int(value) for value in config["linear_calibration_indices"]]
    holdout_windows = [int(value) for value in settings.get("holdout_windows", v2.COMPACT_WINDOW_INDICES)]
    rows_count = 32
    blocks_count = 4
    prepared_cache: dict[int, tuple[Any, dict[tuple[int, str], tuple[Any, dict[str, torch.Tensor]]]]] = {}
    records: list[dict[str, Any]] = []
    state_gate_count = 0
    gate_layers: set[int] = set()
    barrier_witnesses: list[dict[str, Any]] = []

    for layer in layers:
        for role in roles:
            weight_pair = v2._pair(raw.weights[layer][role])
            weight_reference = v2.dequantize_nvfp4(*v2._move_pair(weight_pair, device)).to(torch.float32)
            state, parent_params, artifact = _load_parent_state(
                raw, parent_path, layer, role, args.device, prepared_cache
            )
            parent_params_device = {
                name: value.to(device=device) for name, value in parent_params.items()
            }
            continuous_weight = _state_parent_transform(weight_reference, state)
            parent_weight = ref.dequantize_hif4(
                parent_params_device, tuple(weight_reference.shape)
            ).to(torch.float32)
            fields = _weight_field_views(parent_params, weight_reference.shape, device)
            rows = _uniform_indices(int(weight_reference.shape[0]), rows_count)
            blocks = _uniform_indices(int(weight_reference.shape[1]) // 64, blocks_count)
            folds: list[dict[str, Any]] = []
            for calibration_index in calibration_indices:
                fold = _linear_nvfp4_fold(
                    raw.calibration_activations[role][calibration_index][layer],
                    state, device,
                )
                fold["teacher"] = fold["transformed_reference"].mm(continuous_weight.transpose(0, 1))
                fold["residual"] = fold["teacher"] - fold["quantized"].mm(parent_weight.transpose(0, 1))
                fold["window"] = int(calibration_index)
                folds.append(fold)
            if len(folds) < 2:
                records.append({
                    "layer": layer, "role": role, "status": "SKIPPED_TOO_FEW_FOLDS",
                    "calibration_folds": len(folds), "parent_calibration_artifact": artifact,
                })
                continue

            loo_records = []
            for holdout_index in range(len(folds)):
                train_indices = [index for index in range(len(folds)) if index != holdout_index]
                selected = _select_linear_joint_codes(
                    folds, train_indices, parent_weight, continuous_weight,
                    fields, rows, blocks,
                )
                candidate_residual = _linear_apply_deltas(
                    folds[holdout_index], selected["deltas"], rows, blocks
                )
                gain, parent_mse, candidate_mse = _linear_fold_gain(
                    folds[holdout_index], candidate_residual
                )
                loo_records.append({
                    "held_out_calibration_window": int(folds[holdout_index]["window"]),
                    "train_calibration_windows": [int(folds[index]["window"]) for index in train_indices],
                    "gain": gain, "parent_mse": parent_mse, "candidate_mse": candidate_mse,
                    "attempted_groups": selected["attempted_groups"],
                    "accepted_groups": selected["accepted_groups"],
                    "changed_codes": selected["changed_codes"],
                    "barrier_witness": selected["barrier_witness"],
                })
                if selected["barrier_witness"] is not None:
                    barrier_witnesses.append({
                        "layer": layer, "role": role, "mode": "loo", **selected["barrier_witness"]
                    })

            full_selection = _select_linear_joint_codes(
                folds, list(range(len(folds))), parent_weight, continuous_weight,
                fields, rows, blocks,
            )
            holdout_records = []
            for window in holdout_windows:
                if window < 0 or window >= len(raw.test_windows):
                    raise ValueError(f"R2-L holdout window {window} is outside cache")
                holdout = _linear_holdout_fold(
                    raw, layer, role, window, state, continuous_weight, parent_weight, device
                )
                candidate_residual = _linear_apply_deltas(
                    holdout, full_selection["deltas"], rows, blocks
                )
                gain, parent_mse, candidate_mse = _linear_fold_gain(holdout, candidate_residual)
                holdout_records.append({
                    "window": window, "split": holdout["split"], "length": holdout["length"],
                    "gain": gain, "parent_mse": parent_mse, "candidate_mse": candidate_mse,
                })
            loo_gains = [float(item["gain"]) for item in loo_records]
            holdout_gains = [float(item["gain"]) for item in holdout_records]
            split_means = {
                split: sum(item["gain"] for item in holdout_records if item["split"] == split)
                / max(1, sum(item["split"] == split for item in holdout_records))
                for split in sorted({item["split"] for item in holdout_records})
            }
            state_pass = (
                float(torch.median(torch.tensor(loo_gains))) > 0.0
                and min(loo_gains) >= 0.0
            )
            if state_pass:
                state_gate_count += 1
                gate_layers.add(layer)
            if full_selection["barrier_witness"] is not None:
                barrier_witnesses.append({
                    "layer": layer, "role": role, "mode": "full", **full_selection["barrier_witness"]
                })
            records.append({
                "layer": layer, "role": role, "shape": list(weight_reference.shape),
                "rows": list(rows), "blocks": list(blocks),
                "parent_calibration_artifact": artifact,
                "status": "PASS" if state_pass else "REJECTED",
                "loo": {
                    "records": loo_records,
                    "mean_gain": sum(loo_gains) / max(1, len(loo_gains)),
                    "median_gain": float(torch.median(torch.tensor(loo_gains))),
                    "worst_gain": min(loo_gains),
                },
                "holdout": {
                    "records": holdout_records,
                    "mean_gain": sum(holdout_gains) / max(1, len(holdout_gains)),
                    "median_gain": float(torch.median(torch.tensor(holdout_gains))),
                    "split_mean_gain": split_means,
                    "l1_mean_abs_gain": sum(abs(value) for value in holdout_gains) / max(1, len(holdout_gains)),
                },
                "full_calibration_selection": {
                    "attempted_groups": full_selection["attempted_groups"],
                    "accepted_groups": full_selection["accepted_groups"],
                    "changed_codes": full_selection["changed_codes"],
                    "barrier_witness": full_selection["barrier_witness"],
                },
            })

    focus_records = [item for item in records if item.get("status") in {"PASS", "REJECTED"}]
    loo_passes = sum(item.get("status") == "PASS" for item in focus_records)
    holdout_gains = [
        float(gain)
        for item in focus_records
        for gain in [item["holdout"]["mean_gain"]]
    ]
    holdout_case_gains = [
        float(entry["gain"])
        for item in focus_records
        for entry in item["holdout"]["records"]
    ]
    split_values: dict[str, list[float]] = {}
    for item in focus_records:
        for split, value in item["holdout"]["split_mean_gain"].items():
            split_values.setdefault(split, []).append(float(value))
    split_means = {split: sum(values) / max(1, len(values)) for split, values in split_values.items()}
    l1 = sum(abs(value) for value in holdout_case_gains) / max(1, len(holdout_case_gains))
    g2 = {
        "focus_state_count": len(focus_records),
        "loo_gate_passes": int(loo_passes),
        "loo_gate_required": 8,
        "loo_gate_layers": sorted(gate_layers),
        "loo_gate_layer_count": len(gate_layers),
        "loo_gate_layers_required": 3,
        "holdout_mean_gain": sum(holdout_gains) / max(1, len(holdout_gains)),
        "holdout_median_gain": float(torch.median(torch.tensor(holdout_gains))) if holdout_gains else 0.0,
        "holdout_split_mean_gain": split_means,
        "holdout_l1_mean_abs_case_gain": l1,
        "barrier_witness_count": len(barrier_witnesses),
    }
    g2_pass = (
        loo_passes >= 8
        and len(gate_layers) >= 3
        and g2["holdout_mean_gain"] > 0.0
        and g2["holdout_median_gain"] > 0.0
        and all(value > 0.0 for value in split_means.values())
        and l1 < 0.02
        and bool(barrier_witnesses)
    )
    status = "PASS_TO_R3_B" if g2_pass else "REJECTED"
    expected = {
        "focus_states": len(layers) * len(roles),
        "layers": layers,
        "roles": roles,
        "calibration_folds": calibration_indices,
        "holdout_windows": holdout_windows,
        "rows_per_matrix": rows_count,
        "blocks_per_row": blocks_count,
        "groups_per_block": 8,
        "candidate_patterns_per_group": 256,
    }
    executed = {
        "focus_states": len(focus_records),
        "loo_runs": sum(len(item.get("loo", {}).get("records", [])) for item in records),
        "holdout_cases": len(holdout_case_gains),
        "attempted_groups": sum(item.get("full_calibration_selection", {}).get("attempted_groups", 0) for item in records),
        "accepted_groups": sum(item.get("full_calibration_selection", {}).get("accepted_groups", 0) for item in records),
        "parent_artifacts": sorted({item["parent_calibration_artifact"] for item in records if "parent_calibration_artifact" in item}),
    }
    result = {
        "stage": "r2-linear", "status": status,
        "hypothesis": "固定层级和实际动态量化激活下，8 个连续 mant/sign 码的联合输出目标可能跨折稳定改善 Linear。",
        "expected_cases": expected, "executed_cases": executed,
        "failed_cases": [] if g2_pass else ["G2_L_NOT_SATISFIED"],
        "metrics": {"g2_l": g2, "records": records, "barrier_witnesses": barrier_witnesses,
                    "elapsed_s": time.perf_counter() - started},
        "gate": "G2_L_PASS_TO_R3_B" if g2_pass else "NO_SUPPORTED_DEPLOYABLE_MECHANISM_FROM_R2_L",
        "reason": (
            "R2-L 的 LOO、层覆盖、holdout、split、L1 和 barrier witness 门全部通过，允许进入 R3-B。"
            if g2_pass else
            "R2-L 未同时通过预注册 LOO/holdout/split/L1/barrier 门；固定 8 码分支关闭，不改组大小或扫描邻域。"
        ),
        "next_step": "r3-b" if g2_pass else "r2-attention",
    }
    manifest = _manifest("r2-linear", args, config_path, cache_path, parent_path, status, expected, executed)
    _write_stage(output_dir, result, manifest)
    return result


def _load_parent_attention_state(
    raw: Any, parent_path: Path, layer: int, device: str,
    prepared_cache: dict[int, tuple[Any, dict[int, dict[str, Any]]]],
) -> tuple[dict[str, Any], str]:
    shard = int(layer) % int(v3.SHARD_COUNT)
    if shard not in prepared_cache:
        pack = v3.prepare_shard(raw, shard, "both", ood=False)
        identity = v3._calibration_identity(parent_path.resolve(), pack, torch.device(device))
        artifact = v3.default_calibration_cache_path(identity)
        if not artifact.is_file():
            raise FileNotFoundError(f"verified v186 calibration artifact is missing: {artifact}")
        _, attention_states = v3.load_calibration_artifact(artifact, identity, pack)
        prepared_cache[shard] = (artifact, attention_states)
    artifact, attention_states = prepared_cache[shard]
    return attention_states[int(layer)], str(artifact.resolve())


def _replace_hif4_blocks(
    params: Mapping[str, torch.Tensor], target_rows: Sequence[int], block: int,
    replacement: Mapping[str, torch.Tensor], token_count: int, block_count: int,
) -> dict[str, torch.Tensor]:
    result = {name: value.clone() for name, value in params.items()}
    rows = torch.as_tensor(target_rows, dtype=torch.int64, device=next(iter(result.values())).device)
    for name, value in result.items():
        view = value.reshape(token_count, block_count, *value.shape[2:])
        replacement_view = replacement[name].reshape(len(target_rows), 1, *replacement[name].shape[2:])
        view[rows, int(block)] = replacement_view[:, 0]
    return result


def _attention_oracle_metrics(
    q: torch.Tensor, k: torch.Tensor, value: torch.Tensor,
    q_heads: int, kv_heads: int, head_dim: int,
    reference_output: torch.Tensor, reference_logits: torch.Tensor,
    reference_probabilities: torch.Tensor,
) -> dict[str, float]:
    output, logits, probabilities = v2._attention_trace(
        q[None], k[None], value[None], q_heads, kv_heads, head_dim
    )
    return {
        "output_mse": float((output - reference_output).square().mean()),
        "logit_mse": float((logits - reference_logits).square().mean()),
        "probability_mse": float((probabilities - reference_probabilities).square().mean()),
    }


def run_r2_attention(
    args: argparse.Namespace, config: Mapping[str, Any], config_path: Path,
    output_dir: Path, parent_path: Path, cache_path: Path,
) -> dict[str, Any]:
    started = time.perf_counter()
    raw = v2.load_pack(cache_path)
    device = torch.device(args.device)
    layers = [int(value) for value in config["r1"]["layers"]]
    settings = config["r2_attention"]
    q_head = int(settings["head"])
    kv_head = raw.kv_heads - 1 if str(settings["kv_group"]) == "last" else int(settings["kv_group"])
    token_count = int(settings["query_tokens"])
    attention_cache: dict[int, tuple[Any, dict[int, dict[str, Any]]]] = {}
    records: list[dict[str, Any]] = []
    q_oracle_gains: list[float] = []
    k_oracle_gains: list[float] = []

    for layer in layers:
        states, artifact = _load_parent_attention_state(
            raw, parent_path, layer, args.device, attention_cache
        )
        for calibration_index, item in enumerate(raw.calibration_qkv):
            q_raw, k_raw, v_raw = item[layer]
            q_pair = v2._move_pair(v2._pair(q_raw), device)
            k_pair = v2._move_pair(v2._pair(k_raw), device)
            v_pair = v2._move_pair(v2._pair(v_raw), device)
            q_reference = v2.dequantize_nvfp4(*q_pair).to(torch.float32)
            k_reference = v2.dequantize_nvfp4(*k_pair).to(torch.float32)
            v_reference = v2.dequantize_nvfp4(*v_pair).to(torch.float32)
            q_params = sol.hif4_dynamic_quantize_q(
                q_pair[0], q_pair[1], raw.q_heads, raw.head_dim, states["q_state"]
            )
            k_params = sol.hif4_dynamic_quantize_k(
                k_pair[0], k_pair[1], raw.kv_heads, raw.head_dim, states["k_state"]
            )
            v_params = sol.hif4_dynamic_quantize_v(
                v_pair[0], v_pair[1], raw.kv_heads, raw.head_dim, states["v_state"]
            )
            q_parent = ref.dequantize_hif4(q_params, tuple(q_reference.shape)).to(torch.float32)
            k_parent = ref.dequantize_hif4(k_params, tuple(k_reference.shape)).to(torch.float32)
            v_parent = ref.dequantize_hif4(v_params, tuple(v_reference.shape)).to(torch.float32)
            reference_output, reference_logits, reference_probabilities = v2._attention_trace(
                q_reference[None], k_reference[None], v_reference[None],
                raw.q_heads, raw.kv_heads, raw.head_dim,
            )
            parent_metrics = _attention_oracle_metrics(
                q_parent, k_parent, v_parent, raw.q_heads, raw.kv_heads, raw.head_dim,
                reference_output, reference_logits, reference_probabilities,
            )
            q_rows = _uniform_indices(int(q_reference.shape[0]), token_count)
            k_rows = _uniform_indices(int(k_reference.shape[0]), token_count)

            q_continuous = sol._attention_state_transform_dense(
                q_reference, states["q_state"], raw.q_heads, raw.head_dim, is_k=False
            )
            k_continuous = sol._attention_state_transform_dense(
                k_reference, states["k_state"], raw.kv_heads, raw.head_dim, is_k=True
            )
            q_row_tensor = torch.as_tensor(q_rows, dtype=torch.int64, device=device)
            k_row_tensor = torch.as_tensor(k_rows, dtype=torch.int64, device=device)
            q_exact_params, _ = exact_legal_blocks(
                q_continuous[q_row_tensor, q_head * raw.head_dim:(q_head + 1) * raw.head_dim]
            )
            k_exact_params, _ = exact_legal_blocks(
                k_continuous[k_row_tensor, kv_head * raw.head_dim:(kv_head + 1) * raw.head_dim]
            )
            q_candidate_params = _replace_hif4_blocks(
                q_params, q_rows, q_head, q_exact_params,
                int(q_reference.shape[0]), raw.q_heads,
            )
            k_candidate_params = _replace_hif4_blocks(
                k_params, k_rows, kv_head, k_exact_params,
                int(k_reference.shape[0]), raw.kv_heads,
            )
            q_candidate = ref.dequantize_hif4(
                q_candidate_params, tuple(q_reference.shape)
            ).to(torch.float32)
            k_candidate = ref.dequantize_hif4(
                k_candidate_params, tuple(k_reference.shape)
            ).to(torch.float32)
            q_metrics = _attention_oracle_metrics(
                q_candidate, k_parent, v_parent, raw.q_heads, raw.kv_heads, raw.head_dim,
                reference_output, reference_logits, reference_probabilities,
            )
            k_metrics = _attention_oracle_metrics(
                q_parent, k_candidate, v_parent, raw.q_heads, raw.kv_heads, raw.head_dim,
                reference_output, reference_logits, reference_probabilities,
            )
            q_gain = (parent_metrics["output_mse"] - q_metrics["output_mse"]) / max(parent_metrics["output_mse"], 1.0e-30)
            k_gain = (parent_metrics["output_mse"] - k_metrics["output_mse"]) / max(parent_metrics["output_mse"], 1.0e-30)
            q_oracle_gains.append(q_gain)
            k_oracle_gains.append(k_gain)
            records.append({
                "layer": layer,
                "calibration_window": calibration_index,
                "length": int(q_reference.shape[0]),
                "q_rows": list(q_rows), "k_rows": list(k_rows),
                "q_head": q_head, "kv_head": kv_head,
                "parent": parent_metrics,
                "q_oracle": {**q_metrics, "output_gain": q_gain},
                "k_oracle": {**k_metrics, "output_gain": k_gain},
                "parent_calibration_artifact": artifact,
            })

    g2 = {
        "records": len(records),
        "q_oracle_mean_gain": sum(q_oracle_gains) / max(1, len(q_oracle_gains)),
        "q_oracle_median_gain": float(torch.median(torch.tensor(q_oracle_gains))) if q_oracle_gains else 0.0,
        "k_oracle_mean_gain": sum(k_oracle_gains) / max(1, len(k_oracle_gains)),
        "k_oracle_median_gain": float(torch.median(torch.tensor(k_oracle_gains))) if k_oracle_gains else 0.0,
        "q_oracle_positive": sum(value > 0.0 for value in q_oracle_gains),
        "k_oracle_positive": sum(value > 0.0 for value in k_oracle_gains),
        "deployable_rule": False,
        "oracle_only": True,
    }
    expected = {
        "layers": layers,
        "calibration_folds": len(raw.calibration_windows),
        "head": q_head,
        "kv_group": kv_head,
        "query_tokens_per_tensor": token_count,
        "q_block_replacements_per_record": token_count,
        "k_block_replacements_per_record": token_count,
        "objective": "full Attention output plus logits/probability diagnostics",
    }
    executed = {
        "records": len(records),
        "q_oracle_block_replacements": len(records) * token_count,
        "k_oracle_block_replacements": len(records) * token_count,
        "parent_artifacts": sorted({item["parent_calibration_artifact"] for item in records}),
    }
    result = {
        "stage": "r2-attention", "status": "ORACLE_ONLY",
        "hypothesis": "合法 exact block 替换在真实 GQA Attention 输出上可能有收益，但该替换是否可由在线 API 可见信息确定仍需区分。",
        "expected_cases": expected, "executed_cases": executed,
        "failed_cases": [],
        "metrics": {"g2_a": g2, "records": records, "elapsed_s": time.perf_counter() - started},
        "gate": "ORACLE_ONLY_NO_DEPLOYABLE_RULE",
        "reason": "R2-A 只执行 calibration 上的固定块替换 oracle；它不构成在线联合 Q/K/V 规则，也不允许据此重启已关闭的动态 Gram/importance 族。",
        "next_step": "close-plan",
    }
    manifest = _manifest("r2-attention", args, config_path, cache_path, parent_path, result["status"], expected, executed)
    _write_stage(output_dir, result, manifest)
    return result


def load_config(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("config must be a JSON object")
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage", choices=("r0", "r1", "r2-linear", "r2-attention"), required=True
    )
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser


def main(args: argparse.Namespace | None = None) -> dict[str, Any]:
    args = args or build_parser().parse_args()
    parent_path = args.parent.resolve()
    cache_path = args.cache.resolve()
    config_path = args.config.resolve()
    output_dir = args.output_dir.resolve()
    if not parent_path.is_file():
        raise FileNotFoundError(parent_path)
    if not cache_path.is_file():
        raise FileNotFoundError(cache_path)
    if not config_path.is_file():
        raise FileNotFoundError(config_path)
    config = load_config(config_path)
    if args.stage == "r0":
        return run_r0(args, config, config_path, output_dir, parent_path, cache_path)
    if args.stage == "r1":
        return run_r1(args, config, config_path, output_dir, parent_path, cache_path)
    if args.stage == "r2-linear":
        return run_r2_linear(args, config, config_path, output_dir, parent_path, cache_path)
    return run_r2_attention(args, config, config_path, output_dir, parent_path, cache_path)


if __name__ == "__main__":
    result = main()
    print(json.dumps({"stage": result["stage"], "status": result["status"], "next_step": result["next_step"]}, ensure_ascii=False))
