"""Linear compliance guard: static AST checks + runtime taint checks.

Enforces rule-zero of the 26000 plan §4.4 for ``solution.py``:

Static layer (AST, module-wide):
- rejects the reappearance of known forbidden symbols such as
  ``_linear_output_candidate_metrics``, ``group_cross8``, ``cross8``
  and the ``_ACTIVATION_QUADRATIC8_CROSS_*`` flag family, both as
  identifiers and as string literals (state keys);
- rejects suspicious state keys (output / reference / residual /
  cross / target) in returned calibration dicts;
- flags matmul-style contractions whose two operands mix
  activation-derived and weight-derived names (heuristic for renamed
  equivalents; the runtime layer is authoritative).

Runtime layer (TorchDispatchMode taint tracking):
- seeds the weight input pair with a ``W`` taint and every calibration
  activation pair with an ``A`` taint, propagates taints through every
  ATen op, and tags subtraction outputs with residual suspects
  (``Ra`` / ``Rw``);
- records every contraction (mm / matmul / bmm / einsum family) with
  operand shapes, output shape and combined taints;
- hard failures: contractions combining an activation residual with a
  weight residual (the removed cross8 mechanism in any renamed form),
  contractions leaking the token or out_features dimension into their
  output shape, and activation-state tensors combining both residual
  taints or carrying token/out_features dimensions;
- review entries: dual-taint activation-state tensors that are not
  residual-derived (e.g. the SmoothQuant channel scale) must be
  manually confirmed to be channel-wise statistics.

The guard is evaluator-side only and never feeds anything back into the
solution.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path
from typing import Any, Iterable

import torch

try:  # torch >= 2.0
    from torch.utils._python_dispatch import TorchDispatchMode
except ImportError:  # pragma: no cover
    TorchDispatchMode = None  # type: ignore[assignment]


FORBIDDEN_SYMBOLS: tuple[str, ...] = (
    "_linear_output_candidate_metrics",
    "_activation_cross8_is_safe",
    "_activation_quadratic8_is_safe",
    "group_cross8",
    "cross8",
    "cross64",
)
FORBIDDEN_SYMBOL_PREFIXES: tuple[str, ...] = (
    "_ACTIVATION_QUADRATIC8_CROSS",
    "_LINEAR_OUTPUT",
)
FORBIDDEN_STATE_KEY_PATTERN = re.compile(
    r"(output|reference|residual|cross|target|label)", re.IGNORECASE
)
_CONTRACTION_TOKENS = (
    "mm",
    "matmul",
    "bmm",
    "einsum",
    "addmm",
    "baddbmm",
)
_ACTIVATION_NAMES = (
    "activation",
    "calib",
    "sample",
    "pair",
    "stats",
)
_WEIGHT_NAMES = (
    "weight",
    "dense",
    "w_",
)


def _iter_tensor_args(args: Iterable[Any]) -> Iterable[torch.Tensor]:
    for arg in args:
        if torch.is_tensor(arg):
            yield arg
        elif isinstance(arg, (tuple, list)):
            yield from _iter_tensor_args(arg)


class _TaintRecorder(TorchDispatchMode):  # type: ignore[misc]
    """Record contractions and propagate provenance taints via ATen ops.

    Taint flags:
    - ``A``  — raw activation provenance (seeded on activation pairs);
    - ``G``  — activation Gram provenance (output of an A-only
      contraction, e.g. ``A.T @ A``);
    - ``W``  — weight provenance (seeded on the weight pair; also the
      output taint of contractions involving the weight);
    - ``Ra`` — activation-residual suspect (subtraction involving
      activation-tainted operands);
    - ``Rw`` — weight-residual suspect (subtraction involving
      weight-tainted operands).

    Legal contractions (per the 26000 plan §4.4): ``A.T @ A`` Grams,
    weight-side Grams (``W.T @ W``), the Q(W) Hessian loss
    (weight residual x activation Gram) and the activation refinement
    quadratic form (activation residual x weight Gram).
    Illegal: any contraction combining an activation residual with a
    weight residual (the removed ``cross8`` mechanism), and any
    contraction producing a ``[tokens, out_features]`` output.
    """

    _SUB_TOKENS = ("sub", "subtract", "rsub")

    def __init__(self, in_features: int) -> None:
        super().__init__()
        self.in_features = int(in_features)
        self._taints: dict[int, set[str]] = {}
        self._keepalive: list[Any] = []
        self.contractions: list[dict[str, Any]] = []
        self.cross_residuals: list[dict[str, Any]] = []
        self.op_count = 0

    def seed(self, tensor: torch.Tensor, taint: str) -> None:
        self._taints[id(tensor)] = {taint}
        self._keepalive.append(tensor)

    def taint_of(self, tensor: torch.Tensor) -> set[str]:
        return self._taints.get(id(tensor), set())

    def _register(self, output: Any, taint: set[str]) -> None:
        if torch.is_tensor(output):
            self._taints[id(output)] = set(taint)
            self._keepalive.append(output)
        elif isinstance(output, (tuple, list)):
            for item in output:
                self._register(item, taint)

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):  # noqa: ANN001
        kwargs = kwargs or {}
        self.op_count += 1
        op_name = str(getattr(func, "__name__", func)).lower()
        tensors = list(
            _iter_tensor_args(list(args) + list(kwargs.values()))
        )
        combined: set[str] = set()
        for tensor in tensors:
            combined |= self.taint_of(tensor)

        is_contraction = any(
            token in op_name for token in _CONTRACTION_TOKENS
        )
        if is_contraction:
            entry = {
                "op": op_name,
                "operand_shapes": [tuple(t.shape) for t in tensors],
                "taints": sorted(combined),
                "output_shape": None,
                "cross_residual": (
                    "A" in combined
                    and "W" in combined
                    and "Ra" in combined
                    and "Rw" in combined
                ),
            }
            if entry["cross_residual"]:
                self.cross_residuals.append(entry)
            self.contractions.append(entry)

        output = func(*args, **kwargs)
        if is_contraction and torch.is_tensor(output):
            self.contractions[-1]["output_shape"] = tuple(output.shape)

        if combined:
            out_taint: set[str] = set(combined)
            if is_contraction:
                activation_side = "A" in combined or "Ra" in combined
                if "W" in combined and activation_side:
                    # Weight-guided activation-side contraction (the
                    # legal activation refinement under a weight Gram):
                    # a weaker flag that never counts as weight data.
                    out_taint = {"Wg"} | (
                        {"Ra"} if "Ra" in combined else set()
                    )
                elif "W" in combined:
                    # Pure weight-side contraction (weight Gram, Q(W)
                    # Hessian loss): keep weight residual suspects.
                    out_taint = {"W"} | (
                        {"Rw"} if "Rw" in combined else set()
                    )
                else:
                    # A-only contraction: an activation Gram.
                    out_taint = {"G"} | (
                        {"Ra"} if "Ra" in combined else set()
                    )
            elif (
                len(tensors) >= 2
                and any(token in op_name for token in self._SUB_TOKENS)
            ):
                # A tensor-minus-tensor subtraction is a reconstruction
                # residual only when BOTH operands share the same side
                # (X - X_hat); scalar subtractions and mixed-side subs
                # must not add residual flags.
                operand_taints = [self.taint_of(t) for t in tensors]
                if all("W" in t for t in operand_taints):
                    out_taint.add("Rw")
                if all(("A" in t or "G" in t or "Wg" in t) for t in operand_taints):
                    out_taint.add("Ra")
            if (
                torch.is_tensor(output)
                and output.ndim == 1
                and int(output.shape[0]) == self.in_features
                and not (out_taint & {"Ra", "Rw"})
            ):
                # A [K] channel vector without residual flags is a
                # channel statistic (amax / rms / smooth scale): it
                # cannot carry token or out_features information, and
                # its A/W mixing is the legal SmoothQuant pattern, so
                # it is taint-neutralized.
                out_taint = set()
            self._register(output, out_taint)
        return output


def _collect_names(node: ast.AST) -> set[str]:
    names: set[str] = set()
    for child in ast.walk(node):
        if isinstance(child, ast.Name):
            names.add(child.id.lower())
        elif isinstance(child, ast.Attribute):
            names.add(child.attr.lower())
    return names


def _operand_provenance(node: ast.AST) -> set[str]:
    names = _collect_names(node)
    provenance: set[str] = set()
    for name in names:
        if any(token in name for token in _ACTIVATION_NAMES):
            provenance.add("activation")
        if any(token in name for token in _WEIGHT_NAMES):
            provenance.add("weight")
    return provenance


def _mm_call_nodes(tree: ast.AST):
    for node in ast.walk(tree):
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.MatMult):
            yield node, [node.left, node.right]
        elif isinstance(node, ast.Call):
            name = ""
            operands = list(node.args)
            if isinstance(node.func, ast.Attribute):
                name = node.func.attr.lower()
                # Method call: the receiver (``a`` in ``a.mm(b)``) is an
                # operand too.
                operands.append(node.func.value)
            elif isinstance(node.func, ast.Name):
                name = node.func.id.lower()
            if any(token in name for token in _CONTRACTION_TOKENS):
                yield node, operands


def static_guard(source: str) -> list[str]:
    """Run the static AST compliance checks on solution.py source."""

    violations: list[str] = []
    tree = ast.parse(source)

    for node in ast.walk(tree):
        names: list[str] = []
        if isinstance(node, ast.Name):
            names = [node.id]
        elif isinstance(node, ast.Attribute):
            names = [node.attr]
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            names = [node.value]
        for name in names:
            if name in FORBIDDEN_SYMBOLS or name.startswith(
                FORBIDDEN_SYMBOL_PREFIXES
            ):
                violations.append(f"forbidden symbol/string: {name!r}")

    # Forbidden state keys: only string keys of dict literals (state
    # construction), never error-message strings.
    for node in ast.walk(tree):
        if not isinstance(node, ast.Dict):
            continue
        for key in node.keys:
            if (
                isinstance(key, ast.Constant)
                and isinstance(key.value, str)
                and FORBIDDEN_STATE_KEY_PATTERN.search(key.value)
            ):
                violations.append(f"forbidden state key: {key.value!r}")

    for node, operands in _mm_call_nodes(tree):
        left = _operand_provenance(operands[0]) if operands else set()
        right = (
            _operand_provenance(operands[1]) if len(operands) > 1 else set()
        )
        if ("activation" in left and "weight" in right) or (
            "weight" in left and "activation" in right
        ):
            violations.append(
                "suspected activation/weight cross contraction at line "
                f"{getattr(node, 'lineno', '?')}"
            )
    return violations


def _walk_state(value: Any) -> Iterable[torch.Tensor]:
    if torch.is_tensor(value):
        yield value
    elif isinstance(value, dict):
        for child in value.values():
            yield from _walk_state(child)
    elif isinstance(value, (tuple, list)):
        for child in value:
            yield from _walk_state(child)


def runtime_guard(
    solution: Any,
    weight: torch.Tensor,
    activation_samples: list[torch.Tensor],
    *,
    tokens: int | None = None,
    out_features: int | None = None,
) -> dict[str, Any]:
    """Run linear calibration under taint tracking and audit the result.

    Args:
        solution: loaded solution module.
        weight: ``[M, K]`` dense reference weight operand.
        activation_samples: list of ``[N, K]`` dense calibration activations.
        tokens: expected activation row count (defaults to the first
            sample's row count).
        out_features: expected weight row count (defaults to ``M``).

    Returns:
        A report dict with ``violations`` (hard failures), ``review``
        (items needing human review), contraction statistics, and the
        state tensor inventory.
    """

    if TorchDispatchMode is None:  # pragma: no cover
        raise RuntimeError("TorchDispatchMode is unavailable in this torch build")

    from nvfp4_sim import nvfp4_encode

    weight_pair = nvfp4_encode(weight.detach().to(torch.float32), mode="amax6")
    activation_pairs = [
        nvfp4_encode(sample.detach().to(torch.float32), mode="amax6")
        for sample in activation_samples
    ]
    tokens = int(tokens if tokens is not None else activation_samples[0].shape[0])
    out_features = int(
        out_features if out_features is not None else weight.shape[0]
    )
    in_features = int(weight.shape[1])

    recorder = _TaintRecorder(in_features)
    recorder.seed(weight_pair[0], "W")
    recorder.seed(weight_pair[1], "W")
    for pair in activation_pairs:
        recorder.seed(pair[0], "A")
        recorder.seed(pair[1], "A")

    with recorder:
        result = solution.hif4_calibration_and_quantize_weight(
            weight_pair[0], weight_pair[1], [tuple(pair) for pair in activation_pairs]
        )

    violations: list[str] = []
    review: list[str] = []

    # Rule 1: cross residual contractions (activation residual x weight
    # residual) — the removed cross8 mechanism in any renamed form.
    for entry in recorder.cross_residuals:
        violations.append(
            f"cross residual contraction {entry['op']} with operand "
            f"shapes {entry['operand_shapes']} and taints {entry['taints']}"
        )

    # Rule 2: Linear-output-shaped contractions.  tokens/out_features are
    # chosen as primes that never collide with structural dims; a
    # dual-taint (activation x weight) contraction whose output contains
    # either dimension implies output supervision.  Weight-only
    # contractions may legitimately carry out_features (per-row losses).
    for entry in recorder.contractions:
        output_shape = entry.get("output_shape")
        if output_shape is None:
            continue
        entry_taints = set(entry["taints"])
        if not ("A" in entry_taints and "W" in entry_taints):
            continue
        dims = set(int(dim) for dim in output_shape)
        for probe, label in ((tokens, "token"), (out_features, "out_features")):
            if probe in dims and dims != {probe} and probe > 1:
                violations.append(
                    f"contraction {entry['op']} leaked the {label} "
                    f"dimension in output shape {tuple(output_shape)}"
                )

    # Rule 3: activation state audit.  The weight params legitimately
    # carry the out_features dimension, so only activation_state is
    # audited here (26000 plan §4.6).
    activation_state = (
        result.get("activation_state", {})
        if isinstance(result, dict)
        else {}
    )
    state_tensors = list(_walk_state(activation_state))
    for tensor in state_tensors:
        taint = recorder.taint_of(tensor)
        shape = tuple(int(dim) for dim in tensor.shape)
        taints = set(taint)
        if "A" in taints and "W" in taints:
            if "Ra" in taints and "Rw" in taints:
                violations.append(
                    f"activation state tensor combines activation and "
                    f"weight residuals: shape {shape}"
                )
            else:
                review.append(
                    f"dual-taint activation state tensor with shape "
                    f"{shape} (channel-wise statistic; verify it is not a "
                    f"residual operator)"
                )
        for dim in shape:
            if dim == tokens and tokens > 1:
                violations.append(
                    f"activation state tensor leaks token dimension: "
                    f"shape {shape}"
                )
            elif dim == out_features and out_features > 1:
                violations.append(
                    f"activation state tensor leaks out_features "
                    f"dimension: shape {shape}"
                )

    return {
        "violations": violations,
        "review": review,
        "contraction_count": len(recorder.contractions),
        "contractions": recorder.contractions,
        "aten_op_count": recorder.op_count,
        "state_tensor_count": len(state_tensors),
    }


def guard_solution_file(path: str | Path) -> dict[str, Any]:
    """Static + runtime guard for a solution file (Phase 0 gate entry)."""

    path = Path(path)
    source = path.read_text(encoding="utf-8")
    static_violations = static_guard(source)

    import importlib.util

    spec = importlib.util.spec_from_file_location("_guarded_solution", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    torch.manual_seed(0)
    in_features = 64
    tokens, out_features = 53, 37  # primes: distinct from structural dims
    weight = torch.randn(out_features, in_features) * 0.1
    activations = [
        torch.randn(tokens, in_features) * 0.1 for _ in range(2)
    ]
    report = runtime_guard(
        module,
        weight,
        activations,
        tokens=tokens,
        out_features=out_features,
    )
    report["static_violations"] = static_violations
    report["violations"] = static_violations + report["violations"]
    return report
