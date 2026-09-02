"""Local-only, role-gated one-block output residual probe.

The parent already performs one cross-fold output-weight pass for every
Linear role.  The rejected A3 variant repeated that full pass and added about
78 seconds.  This control asks a narrower question: does one additional
cross-fold output pass on expansive fc matrices, limited to the single highest
leverage 64-channel block, recover the fc regression without the full A3 cost?

The probe imports an immutable parent and never changes the root submission.
"""

from __future__ import annotations

import hashlib
import importlib.util
import sys
import time
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
PARENT_PATH = ROOT / "workbench" / "pre-a3-v147-parent.py"
PARENT_SHA256 = "800ca10ec3414e4fe886b93ca62bd4a350d26bba015287df7e8df2dd871ac23d"
spec = importlib.util.spec_from_file_location("l4_fc_output_parent", PARENT_PATH)
if spec is None or spec.loader is None:
    raise ImportError(f"cannot load source: {PARENT_PATH}")
parent = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = parent
spec.loader.exec_module(parent)
if hashlib.sha256(PARENT_PATH.read_bytes()).hexdigest().lower() != PARENT_SHA256:
    raise RuntimeError("L4 probe parent SHA mismatch")

_LAST_DIAGNOSTIC: dict[str, Any] = {}


def hif4_calibration_and_quantize_weight(
    weight_quant: torch.Tensor,
    weight_scale: torch.Tensor,
    calib_activation_list: list,
) -> dict[str, Any]:
    global _LAST_DIAGNOSTIC
    started = time.perf_counter()
    result = parent.hif4_calibration_and_quantize_weight(
        weight_quant, weight_scale, calib_activation_list
    )
    weight = parent._dequantize_nvfp4_float32(weight_quant, weight_scale).to(torch.float32)
    rows, channels = map(int, weight.shape)
    diagnostic: dict[str, Any] = {
        "expansive": bool(rows > channels),
        "shape": [rows, channels],
        "extra_blocks": 0,
        "changed": False,
    }
    # The current Qwen role layout makes rows > channels the expansive fc
    # family.  Do not touch square/non-expansive roles in this control.
    if rows > channels and channels % int(parent._BLOCK) == 0:
        state = dict(result["activation_state"])
        try:
            weight_t, activation_t = parent._active_transformed_calibration(
                weight_quant, weight_scale, calib_activation_list, state
            )
            # Recompile the activation state after the parent's first output
            # pass, exactly as the historical A3 control did.  Without this
            # refresh the second pass sees stale Q(W) curvature and becomes a
            # misleading no-op.
            deployed_weight = parent._dequantize_hif4(
                result["weight_params"]
            ).to(torch.float32)
            gain = (
                (weight_t * deployed_weight).sum(dim=0)
                / deployed_weight.square().sum(dim=0).clamp_min(parent._EPS)
            )
            gain = torch.nan_to_num(
                gain, nan=1.0, posinf=2.0, neginf=0.5
            ).clamp(0.5, 2.0)
            state["output_gain"] = parent._cpu_tensor(gain)
            state["gram64"] = parent._cpu_tensor(parent._gram64(deployed_weight))
            state["output_cross64"] = parent._cpu_tensor(
                parent._block_cross64(deployed_weight, weight_t.to(torch.float32))
            )
            deployed = parent._active_deploy_from_state(
                result["weight_params"], activation_t, state
            )
            old_count = int(parent._OUTPUT_WEIGHT_HSDQ_BLOCKS)
            old_sweeps = int(parent._OUTPUT_WEIGHT_HSDQ_SWEEPS)
            try:
                parent._OUTPUT_WEIGHT_HSDQ_BLOCKS = 1
                parent._OUTPUT_WEIGHT_HSDQ_SWEEPS = 1
                polished = parent._crossfold_weight_output(
                    weight_t,
                    result["weight_params"],
                    activation_t,
                    deployed,
                )
            finally:
                parent._OUTPUT_WEIGHT_HSDQ_BLOCKS = old_count
                parent._OUTPUT_WEIGHT_HSDQ_SWEEPS = old_sweeps
            before = parent._dequantize_hif4(result["weight_params"]).to(torch.float32)
            after = parent._dequantize_hif4(polished).to(torch.float32)
            changed = bool((before != after).any().item())
            diagnostic["changed"] = changed
            diagnostic["extra_blocks"] = 1 if changed else 0
            diagnostic["changed_values"] = int((before != after).sum().item())
            diagnostic["extra_seconds"] = float(time.perf_counter() - started)
            if changed:
                result["weight_params"] = polished
                deployed_weight = after
                state["gram64"] = parent._cpu_tensor(parent._gram64(deployed_weight))
                state["output_cross64"] = parent._cpu_tensor(
                    parent._block_cross64(deployed_weight, weight_t.to(torch.float32))
                )
                state["output_gain"] = parent._cpu_tensor(
                    (
                        (weight_t * deployed_weight).sum(dim=0)
                        / deployed_weight.square().sum(dim=0).clamp_min(parent._EPS)
                    ).nan_to_num(nan=1.0, posinf=2.0, neginf=0.5).clamp(0.5, 2.0)
                )
                state["l4_fc_one_block"] = True
                state["version"] = 6
                result["activation_state"] = state
        except (RuntimeError, ValueError, FloatingPointError) as exc:
            diagnostic["error"] = f"{type(exc).__name__}: {exc}"
    diagnostic["extra_seconds"] = float(time.perf_counter() - started)
    _LAST_DIAGNOSTIC = diagnostic
    return result


def hif4_dynamic_quantize_activation(
    activation_quant: torch.Tensor,
    activation_scale: torch.Tensor,
    activation_state: Any,
) -> dict[str, torch.Tensor]:
    return parent.hif4_dynamic_quantize_activation(
        activation_quant, activation_scale, activation_state
    )


hif4_calibration_attention = parent.hif4_calibration_attention
hif4_dynamic_quantize_q = parent.hif4_dynamic_quantize_q
hif4_dynamic_quantize_k = parent.hif4_dynamic_quantize_k
hif4_dynamic_quantize_v = parent.hif4_dynamic_quantize_v

__all__ = [
    "hif4_calibration_and_quantize_weight",
    "hif4_dynamic_quantize_activation",
    "hif4_calibration_attention",
    "hif4_dynamic_quantize_q",
    "hif4_dynamic_quantize_k",
    "hif4_dynamic_quantize_v",
]
