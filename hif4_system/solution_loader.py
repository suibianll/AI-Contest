from __future__ import annotations

import hashlib
import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any, Callable


REQUIRED_FUNCTIONS = (
    "hif4_calibration_and_quantize_weight",
    "hif4_dynamic_quantize_activation",
    "hif4_calibration_attention",
    "hif4_dynamic_quantize_q",
    "hif4_dynamic_quantize_k",
    "hif4_dynamic_quantize_v",
)


@dataclass(frozen=True)
class SolutionAPI:
    path: Path
    sha256: str
    module: ModuleType
    functions: dict[str, Callable[..., Any]]

    def function(self, name: str) -> Callable[..., Any]:
        return self.functions[name]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_solution(path: Path) -> SolutionAPI:
    source = path.resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    digest = sha256_file(source)
    module_name = f"_hif4_candidate_{digest[:20]}"
    spec = importlib.util.spec_from_file_location(module_name, source)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot create import spec for {source}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    functions: dict[str, Callable[..., Any]] = {}
    missing: list[str] = []
    for name in REQUIRED_FUNCTIONS:
        value = getattr(module, name, None)
        if not callable(value):
            missing.append(name)
        else:
            functions[name] = value
    if missing:
        raise AttributeError(f"candidate is missing required functions: {', '.join(missing)}")
    return SolutionAPI(source, digest, module, functions)
