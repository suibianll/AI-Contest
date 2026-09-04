"""DIAGNOSTIC-ONLY gate-firing audit wrapper (direction 3, 2026-09-04).

Local diagnostic; never an official submission.  Loads the archived v182
official parent, monkey-patches module-level gate helpers with counters, and
re-exports the six evaluator APIs unchanged.  Counters print via atexit:

- ``_candidate_is_safe`` firing rate, tagged by ``min_mean_improvement``
  (weight block/smooth gates vs attention block gates use different values);
- ``_a1_gate_passes`` pass rate (A1 final-validation gates in calibration);
- ``_fit_attention_pair_matrix_smooth`` non-None rate (pair-transform accept);
- ``_dense_to_hif4`` call profile by ``(max_refine_ratio, max_refine_blocks)``
  (refine-active vs plain encode mix);
- ``_attention_deployed_mse`` / ``_gptq_quantize_weight`` call counts.
"""

import atexit
import importlib.util
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
_TARGET = (
    _ROOT
    / "solutions"
    / "20260904_v182_rank2-linear_v180-attn_scoreNA_timeNA"
    / "solution.py"
)

spec = importlib.util.spec_from_file_location("v182_diag_base", _TARGET)
mod = importlib.util.module_from_spec(spec)
sys.modules["v182_diag_base"] = mod
spec.loader.exec_module(mod)

GATES: dict[str, list[int]] = {}
DENSE_PROFILE: dict[str, int] = {}


def _wrap_bool(name, tag_fn):
    fn = getattr(mod, name)
    rec = GATES.setdefault(name, [0, 0])

    def wrapped(*args, **kwargs):
        rec[0] += 1
        result = fn(*args, **kwargs)
        if result:
            rec[1] += 1
        tag = tag_fn(kwargs) if tag_fn else None
        if tag is not None:
            trec = GATES.setdefault(f"{name}[{tag}]", [0, 0])
            trec[0] += 1
            if result:
                trec[1] += 1
        return result

    return wrapped


def _wrap_optional(name):
    fn = getattr(mod, name)
    rec = GATES.setdefault(name, [0, 0])

    def wrapped(*args, **kwargs):
        rec[0] += 1
        result = fn(*args, **kwargs)
        if result is not None:
            rec[1] += 1
        return result

    return wrapped


mod._candidate_is_safe = _wrap_bool(
    "_candidate_is_safe",
    lambda k: f"mmi={k.get('min_mean_improvement')}",
)
mod._a1_gate_passes = _wrap_bool("_a1_gate_passes", None)
mod._fit_attention_pair_matrix_smooth = _wrap_optional(
    "_fit_attention_pair_matrix_smooth"
)

_dense_orig = mod._dense_to_hif4
_gptq_orig = mod._gptq_quantize_weight
_mse_orig = mod._attention_deployed_mse
DENSE_TOTAL = [0]
GPTQ_CALLS = [0]
MSE_CALLS = [0]


def _dense_logged(*args, **kwargs):
    DENSE_TOTAL[0] += 1
    ratio = float(kwargs.get("max_refine_ratio", 0.0) or 0.0)
    blocks = kwargs.get("max_refine_blocks", None)
    if ratio > 0.0:
        key = f"refine r={ratio:g} b={blocks}"
    else:
        key = "plain"
    DENSE_PROFILE[key] = DENSE_PROFILE.get(key, 0) + 1
    return _dense_orig(*args, **kwargs)


def _gptq_logged(*args, **kwargs):
    GPTQ_CALLS[0] += 1
    return _gptq_orig(*args, **kwargs)


def _mse_logged(*args, **kwargs):
    MSE_CALLS[0] += 1
    return _mse_orig(*args, **kwargs)


mod._dense_to_hif4 = _dense_logged
mod._gptq_quantize_weight = _gptq_logged
mod._attention_deployed_mse = _mse_logged

hif4_calibration_and_quantize_weight = mod.hif4_calibration_and_quantize_weight
hif4_dynamic_quantize_activation = mod.hif4_dynamic_quantize_activation
hif4_calibration_attention = mod.hif4_calibration_attention
hif4_dynamic_quantize_q = mod.hif4_dynamic_quantize_q
hif4_dynamic_quantize_k = mod.hif4_dynamic_quantize_k
hif4_dynamic_quantize_v = mod.hif4_dynamic_quantize_v


@atexit.register
def _dump_diag():
    lines = ["[DIAG] === gate firing audit ==="]
    for key in sorted(GATES):
        calls, accepts = GATES[key]
        rate = (accepts / calls) if calls else 0.0
        lines.append(f"[DIAG] {key}: calls={calls} accept={accepts} rate={rate:.3f}")
    lines.append(f"[DIAG] dense_to_hif4 total={DENSE_TOTAL[0]} gptq={GPTQ_CALLS[0]} deployed_mse={MSE_CALLS[0]}")
    for key in sorted(DENSE_PROFILE):
        lines.append(f"[DIAG] dense[{key}]: {DENSE_PROFILE[key]}")
    print("\n".join(lines), flush=True)
