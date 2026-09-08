"""Build v197 linear-aw1-block-gain: 64-block shared scalar-gain A@W fit.

Appends the L-AW1 post-processing step after the current combined Linear
calibration wrapper.  The parent wrapper is kept intact and aliased; a new
public ``hif4_calibration_and_quantize_weight`` calls it and then runs the
block-gain fit + hard gate + legal re-encode.

solution.py has mixed line endings; the anchor region (combined wrapper tail
at the end of the file) is LF-only, so patching uses \n anchors there.
"""

from pathlib import Path
import hashlib
import json


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
PARENT = ROOT / "solution.py"
CANDIDATE_DIR = HERE / "candidate"
CANDIDATE_DIR.mkdir(parents=True, exist_ok=True)

source = PARENT.read_bytes().decode("utf-8")


def replace_once(old: str, new: str, label: str) -> None:
    global source
    if source.count(old) != 1:
        raise RuntimeError(f"anchor for {label} is not unique; rebuild needs review")
    source = source.replace(old, new, 1)


AW1_CODE = '''


# ---------------------------------------------------------------------------
# v197 L-AW1: 64-block shared scalar-gain A@W low-dimensional fit.
#
# After the parent Linear calibration produces the final deployed weight
# state, fit one scalar gain g_b per 64-wide input block on the official
# normalized output objective
#     L(g) = sum_f ||Y_f - sum_b g_b Z_{f,b}||^2 / (D_f + eps),
#     Z_{f,b} = Ahat_{f,b} @ What_b^T,  D_f = MSE(Y_f, Y_f^STD),
# using sufficient statistics only (no per-fold Z is instantiated):
#     H = sum_f w_f Ahat_f^T Ahat_f,  C = sum_f w_f Ahat_f^T Y_f,
#     G_bc = sum_ij H[b,i,c,j] * (What^T What)[b,i,c,j],
#     h_b  = tr(What_b C_b),        w_f = 1 / (D_f + eps).
# The closed-form ridge solution g* = (G + lambda I)^{-1} h is clamped to
# [0.5, 2.0] and hard-gated against the parent loss L(1) evaluated through
# the same quadratic form; only a strict improvement is deployed, by
# re-encoding g_b * What_b with the root base HiF4 codec and splicing blocks
# (blocks with g_b == 1 or a non-finite re-encode keep the parent codes).
# ---------------------------------------------------------------------------

_AW1_MAX_WINDOWS = 1
_AW1_MAX_TOKENS = 256
_AW1_RIDGE_RATIO = 1.0e-3
_AW1_GAIN_MIN = 0.5
_AW1_GAIN_MAX = 2.0
_AW1_FORCE_IDENTITY = False


@torch.no_grad()
def _aw1_block_gain_fit(
    weight_quant: torch.Tensor,
    weight_scale: torch.Tensor,
    calib_activation_list: list,
    parent_params: dict[str, torch.Tensor],
    state: dict[str, Any],
    diagnostics: dict[str, Any],
) -> Optional[dict[str, torch.Tensor]]:
    weight = _dequantize_nvfp4_float32(weight_quant, weight_scale).to(
        torch.float32
    )
    if weight.ndim != 2:
        return None
    out_features, in_features = map(int, weight.shape)
    if in_features % _HIF4_BLOCK_SIZE != 0:
        return None
    blocks = in_features // _HIF4_BLOCK_SIZE
    w_hat = _dequantize_hif4(parent_params)
    if tuple(w_hat.shape) != (out_features, in_features):
        return None
    w_hat = w_hat.to(torch.float32)
    device = weight.device

    w_std = _dequantize_hif4(_branch_encode_standard_hif4(weight)).to(
        torch.float32
    )

    hessian = torch.zeros(
        in_features, in_features, dtype=torch.float32, device=device
    )
    cross = torch.zeros(
        in_features, out_features, dtype=torch.float32, device=device
    )
    yty = 0.0
    fit_windows = 0
    for pair in calib_activation_list[:_AW1_MAX_WINDOWS]:
        if not isinstance(pair, (tuple, list)) or len(pair) != 2:
            continue
        act_quant = _sample_rows(pair[0], _AW1_MAX_TOKENS)
        act_scale = _sample_rows(pair[1], _AW1_MAX_TOKENS)
        a_fp = _dequantize_nvfp4_float32(act_quant, act_scale).to(torch.float32)
        if a_fp.ndim != 2 or int(a_fp.shape[1]) != in_features:
            continue
        a_hat = _dequantize_hif4(
            hif4_dynamic_quantize_activation(act_quant, act_scale, state)
        ).to(torch.float32)
        a_std = _dequantize_hif4(
            _branch_encode_standard_hif4(a_fp)
        ).to(torch.float32)
        y_ref = a_fp @ weight.t()
        y_std = a_std @ w_std.t()
        denom = float((y_ref - y_std).square().mean())
        omega = 1.0 / (denom + _EPS)
        hessian.add_(a_hat.t() @ a_hat, alpha=omega)
        cross.add_(a_hat.t() @ y_ref, alpha=omega)
        yty += omega * float(y_ref.square().sum())
        fit_windows += 1
    diagnostics["aw1_fit_windows"] = int(fit_windows)
    if fit_windows == 0:
        return None
    diagnostics["aw1_attempted"] = 1
    diagnostics["aw1_blocks"] = int(blocks)

    w_gram = w_hat.t() @ w_hat
    h4 = hessian.reshape(
        blocks, _HIF4_BLOCK_SIZE, blocks, _HIF4_BLOCK_SIZE
    )
    s4 = w_gram.reshape(
        blocks, _HIF4_BLOCK_SIZE, blocks, _HIF4_BLOCK_SIZE
    )
    gram_g = (h4 * s4).sum(dim=(1, 3))
    rhs = (
        w_hat.t().reshape(blocks, _HIF4_BLOCK_SIZE, out_features)
        * cross.reshape(blocks, _HIF4_BLOCK_SIZE, out_features)
    ).sum(dim=(1, 2))

    g64 = gram_g.to(torch.float64)
    h64 = rhs.to(torch.float64)
    ridge = _AW1_RIDGE_RATIO * max(float(g64.diagonal().mean()), 0.0) + 1.0e-30
    eye = torch.eye(blocks, dtype=torch.float64, device=device)
    gains = torch.linalg.solve(g64 + ridge * eye, h64)
    gains = torch.nan_to_num(
        gains, nan=1.0, posinf=_AW1_GAIN_MAX, neginf=_AW1_GAIN_MIN
    )
    gains = gains.clamp(min=_AW1_GAIN_MIN, max=_AW1_GAIN_MAX)
    if _AW1_FORCE_IDENTITY:
        gains = torch.ones_like(gains)

    def _quad_loss(g: torch.Tensor) -> float:
        return yty - 2.0 * float(g @ h64) + float(g @ (g64 @ g))

    loss_one = _quad_loss(torch.ones_like(h64))
    loss_gain = _quad_loss(gains)
    diagnostics["aw1_loss_parent"] = float(loss_one)
    diagnostics["aw1_loss_candidate"] = float(loss_gain)
    if not (math.isfinite(loss_gain) and math.isfinite(loss_one)):
        return None
    if not loss_gain < loss_one:
        return None

    gains32 = gains.to(torch.float32)
    diagnostics["aw1_gain_abs_dev_mean"] = float((gains32 - 1.0).abs().mean())
    scaled = w_hat * gains32.repeat_interleave(_HIF4_BLOCK_SIZE).reshape(
        1, in_features
    )
    candidate_params = _dense_to_hif4(scaled)
    decoded = _dequantize_hif4(candidate_params).reshape(
        out_features, blocks, _HIF4_BLOCK_SIZE
    )
    block_finite = torch.isfinite(decoded).all(dim=0).all(dim=-1)
    keep_parent = (gains32 == 1.0).to(device=decoded.device) | (~block_finite)
    if bool(keep_parent.all()):
        return None
    merged: dict[str, torch.Tensor] = {}
    for key, parent_value in parent_params.items():
        candidate_value = candidate_params.get(key)
        if (
            not torch.is_tensor(parent_value)
            or not torch.is_tensor(candidate_value)
            or tuple(candidate_value.shape) != tuple(parent_value.shape)
            or candidate_value.dtype != parent_value.dtype
        ):
            return None
        mask = keep_parent.reshape(1, blocks, 1, 1, 1).to(
            device=parent_value.device
        )
        merged[key] = torch.where(
            mask, parent_value, candidate_value.to(parent_value.device)
        )
    diagnostics["aw1_accepted"] = 1
    diagnostics["aw1_kept_parent_blocks"] = int(keep_parent.sum())
    diagnostics["aw1_gains"] = gains32.detach().to(
        device="cpu", dtype=torch.float32
    ).contiguous()
    return merged


@torch.no_grad()
def _aw1_block_gain_postprocess(
    weight_quant: torch.Tensor,
    weight_scale: torch.Tensor,
    calib_activation_list: list,
    result: dict[str, Any],
) -> dict[str, Any]:
    state = result.get("activation_state") if isinstance(result, dict) else None
    parent_params = result.get("weight_params") if isinstance(result, dict) else None
    diagnostics: dict[str, Any] = {
        "aw1_attempted": 0,
        "aw1_accepted": 0,
        "aw1_arm": "unavailable",
    }
    if not isinstance(state, dict) or not isinstance(parent_params, dict):
        return result
    updated = None
    try:
        updated = _aw1_block_gain_fit(
            weight_quant,
            weight_scale,
            calib_activation_list,
            parent_params,
            state,
            diagnostics,
        )
    except (RuntimeError, ValueError, TypeError, IndexError, KeyError):
        updated = None
    diagnostics["aw1_arm"] = "accepted" if updated is not None else (
        "parent" if diagnostics.get("aw1_attempted") else "unavailable"
    )
    state.update(diagnostics)
    if updated is None:
        return result
    return dict(result, weight_params=updated)


_AW1_PARENT_LINEAR_CALIBRATION = hif4_calibration_and_quantize_weight


@torch.no_grad()
def hif4_calibration_and_quantize_weight(
    weight_quant: torch.Tensor,
    weight_scale: torch.Tensor,
    calib_activation_list: list,
) -> dict[str, Any]:
    """v197: parent combined Linear calibration + AW1 block-gain postprocess."""

    result = _AW1_PARENT_LINEAR_CALIBRATION(
        weight_quant, weight_scale, calib_activation_list
    )
    return _aw1_block_gain_postprocess(
        weight_quant, weight_scale, calib_activation_list, result
    )
'''

# Anchor: the tail of the combined Linear calibration wrapper (unique because
# of the COMBINED-LINEAR-SAMPLE-ENERGY print) plus the start of the next def.
anchor = "\n".join([
    "    print(",
    '        f"[COMBINED-LINEAR-SAMPLE-ENERGY] reachable=1 blocks={int(order.numel())}",',
    "        flush=True,",
    "    )",
    "    return result",
    "",
    "",
    "@torch.no_grad()",
    "def _combined_dynamic_sample_energy_block_order_fast(",
])
replacement = "\n".join([
    "    print(",
    '        f"[COMBINED-LINEAR-SAMPLE-ENERGY] reachable=1 blocks={int(order.numel())}",',
    "        flush=True,",
    "    )",
    "    return result",
]) + AW1_CODE + "\n\n@torch.no_grad()\ndef _combined_dynamic_sample_energy_block_order_fast("
replace_once(anchor, replacement, "aw1-insert")

candidate = CANDIDATE_DIR / "solution.py"
candidate.write_bytes(source.encode("utf-8"))

config = {
    "run_id": "linear-aw1-block-gain",
    "version": "v197",
    "mechanism": "L-AW1: 64-block shared scalar-gain A@W closed-form ridge fit on the official normalized output loss, hard-gated, compiled by re-encoding g_b*What_b with the base HiF4 codec",
    "parent": "solution.py",
    "parent_sha256": hashlib.sha256(PARENT.read_bytes()).hexdigest(),
    "source_sha256": hashlib.sha256(candidate.read_bytes()).hexdigest(),
    "config": {
        "max_windows": 1,
        "max_tokens_per_window": 256,
        "ridge_ratio": 1.0e-3,
        "gain_clamp": [0.5, 2.0],
    },
    "candidate_count": 1,
    "official_status": "unregistered/NA",
}
(HERE / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
print(json.dumps(config, indent=2))
