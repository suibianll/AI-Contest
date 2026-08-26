from __future__ import annotations

import ast
import copy
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from .solution_loader import REQUIRED_FUNCTIONS


@dataclass(frozen=True)
class ComplianceReport:
    path: Path
    violations: tuple[str, ...]
    warnings: tuple[str, ...] = ()

    @property
    def passed(self) -> bool:
        return not self.violations


class _Visitor(ast.NodeVisitor):
    def __init__(self) -> None:
        self.violations: set[str] = set()
        self.functions: set[str] = set()
        self._function_stack: list[str] = []

    @property
    def current_function(self) -> str:
        return self._function_stack[-1] if self._function_stack else ""

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self.functions.add(node.name)
        self._function_stack.append(node.name)
        self.generic_visit(node)
        self._function_stack.pop()

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_Import(self, node: ast.Import) -> None:
        if any(alias.name == "numpy" or alias.name.startswith("numpy.") for alias in node.names):
            self.violations.add("numpy_import")
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if node.module and (node.module == "numpy" or node.module.startswith("numpy.")):
            self.violations.add("numpy_import")
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        name = node.func.id if isinstance(node.func, ast.Name) else getattr(node.func, "attr", "")
        if name in {"open", "read_text", "write_text", "read_bytes", "write_bytes", "unlink"}:
            self.violations.add("file_io")
        if name in {"matmul", "mm", "bmm", "einsum"} and any(
            token in self.current_function.lower() for token in ("weight", "activation", "linear")
        ):
            self.violations.add("linear_output_fit")
        self.generic_visit(node)

    def visit_BinOp(self, node: ast.BinOp) -> None:
        if isinstance(node.op, ast.MatMult) and any(
            token in self.current_function.lower() for token in ("weight", "activation", "linear")
        ):
            self.violations.add("linear_output_fit")
        self.generic_visit(node)


def check_static(path: Path) -> ComplianceReport:
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))
    visitor = _Visitor()
    visitor.visit(tree)
    missing = set(REQUIRED_FUNCTIONS) - visitor.functions
    visitor.violations.update(f"missing_interface:{name}" for name in sorted(missing))
    return ComplianceReport(path.resolve(), tuple(sorted(visitor.violations)))


_ALLOWED_TENSOR_DTYPES = {
    torch.bool,
    torch.int8,
    torch.int16,
    torch.int32,
    torch.int64,
    torch.float16,
    torch.bfloat16,
    torch.float32,
}


def validate_state(value: Any, *, max_depth: int = 8, max_nodes: int = 4096) -> None:
    nodes = 0
    active: set[int] = set()

    def visit(item: Any, depth: int) -> None:
        nonlocal nodes
        nodes += 1
        if nodes > max_nodes:
            raise ValueError("state node count exceeds 4096")
        if depth > max_depth:
            raise ValueError("state depth exceeds 8")
        if item is None or isinstance(item, (bool, int)):
            return
        if isinstance(item, str):
            if len(item.encode("utf-8")) > 4096:
                raise ValueError("state strings must not exceed 4096 UTF-8 bytes")
            return
        if isinstance(item, float):
            if not math.isfinite(item):
                raise ValueError("state float must be finite")
            return
        if torch.is_tensor(item):
            if item.device.type != "cpu":
                raise ValueError("state tensors must be CPU tensors")
            if item.layout != torch.strided:
                raise ValueError("state tensors must use dense strided layout")
            if item.requires_grad or item.is_complex() or item.dtype not in _ALLOWED_TENSOR_DTYPES:
                raise ValueError("state tensor has an unsupported dtype or gradient")
            if item.is_floating_point() and not bool(torch.isfinite(item).all()):
                raise ValueError("state tensor must be finite")
            return
        if isinstance(item, dict):
            identity = id(item)
            if identity in active:
                raise ValueError("state contains a cycle")
            active.add(identity)
            try:
                for key, child in item.items():
                    if not isinstance(key, str):
                        raise ValueError("state dict keys must be strings")
                    if len(key.encode("utf-8")) > 4096:
                        raise ValueError("state dict keys must not exceed 4096 UTF-8 bytes")
                    visit(child, depth + 1)
            finally:
                active.remove(identity)
            return
        if isinstance(item, (list, tuple)):
            identity = id(item)
            if identity in active:
                raise ValueError("state contains a cycle")
            active.add(identity)
            try:
                for child in item:
                    visit(child, depth + 1)
            finally:
                active.remove(identity)
            return
        raise ValueError(f"state contains unsupported type: {type(item).__name__}")

    visit(value, 0)


def clone_state(value: Any) -> Any:
    """Return the independent online-test state required by the official runner."""

    validate_state(value)
    return copy.deepcopy(value)
