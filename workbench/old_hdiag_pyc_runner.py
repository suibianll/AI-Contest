"""Diagnostic runner for the preserved static-actorder bytecode."""

import importlib.machinery
import importlib.util
import sys
from pathlib import Path


_CACHE = Path(__file__).resolve().parent / "__pycache__"
_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))


def _load_sourceless(name: str):
    path = _CACHE / f"{name}.cpython-312.pyc"
    loader = importlib.machinery.SourcelessFileLoader(name, str(path))
    spec = importlib.util.spec_from_loader(name, loader)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    loader.exec_module(module)
    return module


_load_sourceless("linear_static_actorder_solution")
_IMPL = _load_sourceless("linear_static_actorder_hdiag_solution")

for _name in (
    "hif4_calibration_and_quantize_weight",
    "hif4_dynamic_quantize_activation",
    "hif4_calibration_attention",
    "hif4_dynamic_quantize_q",
    "hif4_dynamic_quantize_k",
    "hif4_dynamic_quantize_v",
):
    globals()[_name] = getattr(_IMPL, _name)
