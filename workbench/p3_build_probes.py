"""Generate the P3 official-contribution probe single files (plan section 7).

P3-A (path effects, fixed standard Linear), built on the v164 archive
(v160 Attention path + standard Linear, official 13945):
  A10 - Q/K keep the v160 path, V swapped to the standard codec. New submit.
  A01 - Q/K swapped to the standard codec, V keeps the v160 path. New submit.
(A00 = v162 archive, A11 = v164 archive; both reused, not rebuilt.)

P3-C (Linear shape buckets, fixed standard Attention), built on the v163
archive (v160 Linear + standard Attention, official 4587). Four submits, one
per bucket; only weights whose [rows, cols] land in the target bucket keep the
v160 Linear path, everything else falls back to the standard codec:
  W0 rows <= 256
  W1 rows > 256 and 0.75 <= rows/cols <= 1.33
  W2 rows > 256 and rows/cols > 1.33
  W3 rows > 256 and rows/cols < 0.75 (remainder)

P3-B (A1 length buckets) is DESIGN_BLOCKED: the official contract exposes Q
and K as independent API calls with no shared scene key, so length-bucket
switching cannot be made coordinate-safe.

Probe files are exact copies of their base archive plus an appended redef
block, so the untouched paths are bit-identical by construction.
"""

from __future__ import annotations

from pathlib import Path

import torch  # noqa: F401  (mirrors the module-level torch import in archives)

ROOT = Path(__file__).resolve().parents[1]
SOL = ROOT / "solutions"

V162 = SOL / "20260903_v162_standard-baseline-both_scoreNA_timeNA" / "solution.py"
V163 = SOL / "20260903_v163_v160-linear_standard-attn_scoreNA_timeNA" / "solution.py"
V164 = SOL / "20260903_v164_standard-linear_v160-attn_scoreNA_timeNA" / "solution.py"

A10_APPENDIX = '''

# ---------------------------------------------------------------------------
# P3-A probe A10: Q/K keep the v160 path, V swapped to the standard codec.
# Appended redefinitions override the v160 hif4_dynamic_quantize_v; the v160
# Q/K dynamic functions, the v160 attention calibration, and the standard
# Linear path above are untouched.  The V state produced by calibration is
# legal but unused by the standard V encoder.
# ---------------------------------------------------------------------------


@torch.no_grad()
def hif4_dynamic_quantize_v(
    v_quant: torch.Tensor,
    v_scale: torch.Tensor,
    kv_num_heads: int,
    head_dim: int,
    v_state: Any,
) -> dict[str, torch.Tensor]:
    return _ref_encode_standard_hif4(
        _dequantize_nvfp4_float32(v_quant, v_scale)
    )
'''

A01_APPENDIX = '''

# ---------------------------------------------------------------------------
# P3-A probe A01: Q/K swapped to the standard codec, V keeps the v160 path.
# Appended redefinitions override the v160 hif4_dynamic_quantize_q and
# hif4_dynamic_quantize_k; the v160 V dynamic function, the v160 attention
# calibration, and the standard Linear path above are untouched.  The Q/K
# states produced by calibration are legal but unused.
# ---------------------------------------------------------------------------


@torch.no_grad()
def hif4_dynamic_quantize_q(
    q_quant: torch.Tensor,
    q_scale: torch.Tensor,
    q_num_heads: int,
    head_dim: int,
    q_state: Any,
) -> dict[str, torch.Tensor]:
    return _ref_encode_standard_hif4(
        _dequantize_nvfp4_float32(q_quant, q_scale)
    )


@torch.no_grad()
def hif4_dynamic_quantize_k(
    k_quant: torch.Tensor,
    k_scale: torch.Tensor,
    kv_num_heads: int,
    head_dim: int,
    k_state: Any,
) -> dict[str, torch.Tensor]:
    return _ref_encode_standard_hif4(
        _dequantize_nvfp4_float32(k_quant, k_scale)
    )
'''

_BUCKET_TEMPLATE = '''

# ---------------------------------------------------------------------------
# P3-C probe __NAME__: only the __BUCKET__ weight bucket keeps the v160 Linear
# path; every other bucket uses the standard HiF4 Linear codec.  Attention is
# standard (v163 archive).  Appended redefinitions wrap the two Linear APIs.
# ---------------------------------------------------------------------------

_P3_TARGET_BUCKET = "__BUCKET__"


def _p3_bucket_of(rows: int, cols: int) -> str:
    if rows <= 256:
        return "W0"
    ratio = rows / cols
    if 0.75 <= ratio <= 1.33:
        return "W1"
    if ratio > 1.33:
        return "W2"
    return "W3"


_p3_orig_weight_calibration = hif4_calibration_and_quantize_weight
_p3_orig_activation = hif4_dynamic_quantize_activation


@torch.no_grad()
def hif4_calibration_and_quantize_weight(
    weight_quant: torch.Tensor,
    weight_scale: torch.Tensor,
    calib_activation_list: list,
) -> dict[str, Any]:
    shape = _dequantize_nvfp4_float32(weight_quant, weight_scale).shape
    if _p3_bucket_of(int(shape[0]), int(shape[1])) == _P3_TARGET_BUCKET:
        return _p3_orig_weight_calibration(
            weight_quant, weight_scale, calib_activation_list
        )
    return {
        "weight_params": _ref_encode_standard_hif4(
            _dequantize_nvfp4_float32(weight_quant, weight_scale)
        ),
        "activation_state": {},
    }


@torch.no_grad()
def hif4_dynamic_quantize_activation(
    activation_quant: torch.Tensor,
    activation_scale: torch.Tensor,
    activation_state: Any,
) -> dict[str, torch.Tensor]:
    if activation_state:
        return _p3_orig_activation(
            activation_quant, activation_scale, activation_state
        )
    return _ref_encode_standard_hif4(
        _dequantize_nvfp4_float32(activation_quant, activation_scale)
    )
'''


def build() -> None:
    v164 = V164.read_text(encoding="utf-8")
    v163 = V163.read_text(encoding="utf-8")
    targets = [
        (
            "20260905_p3a_a10_qk-v160_v-std_probe",
            v164,
            A10_APPENDIX,
        ),
        (
            "20260905_p3a_a01_qk-std_v-v160_probe",
            v164,
            A01_APPENDIX,
        ),
    ]
    for bucket, name_suffix in (("W0", "w0"), ("W1", "w1"), ("W2", "w2"), ("W3", "w3")):
        appendix = (
            _BUCKET_TEMPLATE.replace("__BUCKET__", bucket)
            .replace("__NAME__", f"W{bucket[1]}")
        )
        targets.append((f"20260905_p3c_{name_suffix}_linear-bucket_probe", v163, appendix))

    for directory, base, appendix in targets:
        out_dir = SOL / directory
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / "solution.py"
        path.write_text(base + appendix, encoding="utf-8")
        print(f"wrote {path} ({len(base) + len(appendix)} chars)")


if __name__ == "__main__":
    build()
