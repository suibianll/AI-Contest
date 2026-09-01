"""Local composition control: v140 Linear with v86 Attention.

This file deliberately delegates the six public APIs to the two immutable
archives.  It is a reproducible attribution experiment, not an official
submission snapshot: the official package expects a self-contained solution
file, while this control is used to measure the effect of restoring v86's
Attention on top of the already archived v140 Linear path.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType


_ROOT = Path(__file__).resolve().parents[2]
_V86_PATH = (
    _ROOT
    / "solutions"
    / "20260830_v086_c86-attn-block-final_scoreNA_timeNA"
    / "solution.py"
)
_V140_PATH = (
    _ROOT
    / "solutions"
    / "20260901_v140_linear-roab-pair_rejected"
    / "solution.py"
)


def _load_archive(name: str, path: Path) -> ModuleType:
    if not path.is_file():
        raise FileNotFoundError(f"archive source is missing: {path}")
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load archive source: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_v86 = _load_archive("_hif4_v86_attention", _V86_PATH)
_v140 = _load_archive("_hif4_v140_linear", _V140_PATH)


# Keep the v140 Linear implementation and the v86 Attention implementation
# completely separate.  Their states use only the public evaluator contract.
hif4_calibration_and_quantize_weight = _v140.hif4_calibration_and_quantize_weight
hif4_dynamic_quantize_activation = _v140.hif4_dynamic_quantize_activation
hif4_calibration_attention = _v86.hif4_calibration_attention
hif4_dynamic_quantize_q = _v86.hif4_dynamic_quantize_q
hif4_dynamic_quantize_k = _v86.hif4_dynamic_quantize_k
hif4_dynamic_quantize_v = _v86.hif4_dynamic_quantize_v


__all__ = (
    "hif4_calibration_and_quantize_weight",
    "hif4_dynamic_quantize_activation",
    "hif4_calibration_attention",
    "hif4_dynamic_quantize_q",
    "hif4_dynamic_quantize_k",
    "hif4_dynamic_quantize_v",
)
