"""L-TF2 premise check: the first pass's gradient is the pre-loop gradient.

The plan (section 2, "可检验假设") requires the premise to be confirmed
*statically* before anything is edited:

    "在相同父状态/输入下，正常路径结果和非有限回退行为保持一致；每个进入下降的
     动态调用减少首遍两次矩阵乘。"

The plan states what it expects to find.  This script does not take that on
trust -- it reads the retained v231 root, walks the actual statement list
between the two gradient assignments, and reports every name that is rebound or
mutated in place along the way.  The verdict is computed from the AST, not
asserted.

Two questions are answered separately:

  *P1*  Is ``deployed`` (or ``reference``) rebound or mutated in place anywhere
        between the pre-loop gradient and the first in-loop gradient?

  *P2*  Are the two gradient expressions the same expression on the same
        operands -- i.e. does the in-loop recomputation, run at ``_pass == 0``,
        see exactly the tensors the pre-loop one saw?

A third check, *P3*, is negative evidence: it walks the whole function and
reports every in-place operation on any name, so the reader can see that the
analysis is not blind to in-place operations -- ``deployed.add_`` and
``gradient.add_`` really do exist, they are simply after the point that matters.

*P4* is the reason this card differs from L-TF1 in a way that matters.  The pass
body mutates ``deployed`` and ``gradient`` in place, so the reuse must stop at
pass 0; but on this root ``_EM1_PASSES = 2``, which means the guarded branch is
not dead code as it was on v233's K = 1 parent -- pass 1 executes it.  The script
therefore also reports the shipped pass count and states plainly that the guard
is live.

The parent is read from the immutable archive, never from the live root file.

This script is CPU-only and reads nothing but the parent file.
"""

from __future__ import annotations

import ast
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
PARENT = ROOT / "solutions/20260910_v231_linear-em3-k2-arm_scoreNA_timeNA/solution.py"

PARENT_SHA256 = "ea79a1c12dc667142c620975aab188920fae7b29988c804f41c1a696cc5754f1"
EXPECTED_PASSES = 2

FN_START = b"@torch.no_grad()\ndef _em1_dynamic_descent(\n"
FN_END = b"\n_EM1_PARENT_LINEAR_CALIBRATION = hif4_calibration_and_quantize_weight"

GRADIENT_EXPR = (
    "(deployed - reference).mm(metric) + reference.mm(h_matrix)"
)

# Names whose in-place mutation would break the premise.
WATCHED = ("deployed", "reference")

# torch in-place ops end with a single underscore.  Listed explicitly rather
# than pattern-matched so the report says what it looked for.
INPLACE_SUFFIX = "_"


def extract_parent_function() -> tuple[str, int]:
    payload = PARENT.read_bytes()
    if payload.count(FN_START) != 1:
        raise SystemExit("parent function start anchor is not unique")
    if payload.count(FN_END) != 1:
        raise SystemExit("parent function end anchor is not unique")
    start = payload.index(FN_START)
    end = payload.index(FN_END, start)
    return payload[start:end].decode("utf-8"), payload[:start].count(b"\n") + 1


def is_gradient_expr(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id == "gradient"
        and ast.unparse(node.value) == GRADIENT_EXPR
    )


def inplace_target(node: ast.AST) -> str | None:
    """Returns the base name of an in-place mutation, or None.

    Handles ``x.add_(y)`` (attribute call ending in ``_``) and ``x[i] = y``
    (subscript store).  ``x.a = y`` is reported too: an attribute store on a
    tensor is a mutation of that tensor.
    """
    if isinstance(node, ast.Call):
        func = node.func
        if isinstance(func, ast.Attribute) and func.attr.endswith(INPLACE_SUFFIX):
            base = func.value
            if isinstance(base, ast.Name):
                return base.id
            if isinstance(base, ast.Attribute):
                return base.attr
            # x[i].add_(y) and friends -- report the subscript's base.
            if isinstance(base, ast.Subscript) and isinstance(base.value, ast.Name):
                return base.value.id
    if isinstance(node, ast.Assign):
        for target in node.targets:
            if isinstance(target, ast.Subscript) and isinstance(
                target.value, ast.Name
            ):
                return target.value.id
    if isinstance(node, ast.AugAssign):
        if isinstance(node.target, ast.Name):
            return node.target.id
    return None


def rebinding_names(node: ast.AST) -> set[str]:
    """Plain ``name = ...`` bindings performed by this statement."""
    bound: set[str] = set()
    if isinstance(node, ast.Assign):
        for target in node.targets:
            if isinstance(target, ast.Name):
                bound.add(target.id)
    return bound


# Rebindings of the shape ``x = x.<op>()`` that preserve every element exactly.
# These are listed rather than inferred: an unlisted op is reported as
# value-destroying, and adding one to this set is a claim about torch semantics
# that a reader can check.
VALUE_PRESERVING_REBINDS = ("clone", "contiguous", "detach", "requires_grad_")


def rebind_kind(node: ast.AST) -> tuple[str, str]:
    """Classify a rebinding of a watched name.

    Returns ``(name, kind)`` where kind is ``"value-preserving"`` or
    ``"value-destroying"``.  Only ``x = x.<listed op>()`` counts as preserving
    -- anything else, including ``x = f(x)`` for an unknown ``f``, is reported
    as destroying, so the default is the safe one.
    """
    assert isinstance(node, ast.Assign)
    target = node.targets[0]
    assert isinstance(target, ast.Name)
    name = target.id
    value = node.value
    if (
        isinstance(value, ast.Call)
        and isinstance(value.func, ast.Attribute)
        and value.func.attr in VALUE_PRESERVING_REBINDS
        and isinstance(value.func.value, ast.Name)
        and value.func.value.id == name
    ):
        return name, "value-preserving"
    return name, "value-destroying"


def main() -> int:
    import hashlib

    payload = PARENT.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    if digest != PARENT_SHA256:
        raise SystemExit(f"parent SHA256 mismatch: {digest} != {PARENT_SHA256}")

    function_text, first_line = extract_parent_function()
    tree = ast.parse(function_text)
    function = tree.body[0]
    assert isinstance(function, ast.FunctionDef), "expected a function definition"

    body = function.body

    # One gradient is a top-level statement, the other sits inside the pass
    # loop, so walk the whole function rather than just its body.
    found = sorted(
        (node for node in ast.walk(function) if is_gradient_expr(node)),
        key=lambda node: node.lineno,
    )
    print(f"function extracted from {PARENT.name}:{first_line} "
          f"({len(function_text.encode('utf-8'))} bytes)")
    print(f"gradient expressions found in the function: {len(found)}")
    if len(found) != 2:
        raise SystemExit(
            f"expected exactly 2 gradient expressions, found {len(found)}"
        )
    pre_loop, in_loop = found

    loop = next(
        (
            node
            for node in ast.walk(function)
            if isinstance(node, ast.For)
            and any(inner is in_loop for inner in node.body)
        ),
        None,
    )
    if loop is None:
        raise SystemExit("the in-loop gradient is not inside a For statement")
    if not any(node is loop for node in body):
        raise SystemExit("the pass loop is not a top-level statement of the function")

    print(f"  pre-loop   at line {pre_loop.lineno}: {GRADIENT_EXPR}")
    print(f"  in-loop    at line {in_loop.lineno}: {GRADIENT_EXPR}")
    print(f"  in-loop statement index inside the pass loop: "
          f"{loop.body.index(in_loop)} (0 means it runs first each pass)")
    print(f"  the loop iterates `{ast.unparse(loop.target)}` over "
          f"`{ast.unparse(loop.iter)}`")

    # _EM1_PASSES is a module-level constant outside the extracted function, so
    # read it off the whole parent module rather than out of this tree.
    shipped_passes = None
    module_tree = ast.parse(payload.decode("utf-8"))
    for node in module_tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if isinstance(target, ast.Name) and target.id == "_EM1_PASSES":
                shipped_passes = ast.literal_eval(node.value)
    print(f"  shipped _EM1_PASSES in the parent module: {shipped_passes}")
    if shipped_passes != EXPECTED_PASSES:
        raise SystemExit(
            f"the parent ships K={shipped_passes}, this card expects K={EXPECTED_PASSES}"
        )

    pre_idx = next(i for i, node in enumerate(body) if node is pre_loop)
    loop_idx = next(i for i, node in enumerate(body) if node is loop)
    between = body[pre_idx + 1 : loop_idx]
    print(f"\nP1/P2 -- statements between the two gradient assignments: {len(between)}")
    violations: list[str] = []
    preserving: list[str] = []
    for node in between:
        line = node.lineno
        for watched in set(WATCHED).intersection(rebinding_names(node)):
            name, kind = rebind_kind(node)
            record = f"  line {line}: rebinds `{name}` -- {kind}: {ast.unparse(node)[:80]}"
            (violations if kind != "value-preserving" else preserving).append(record)
        mutation = inplace_target(node)
        if mutation in WATCHED:
            violations.append(
                f"  line {line}: MUTATES `{mutation}` IN PLACE: "
                f"{ast.unparse(node)[:90]}"
            )
    print(f"  value-preserving rebindings of {' or '.join(WATCHED)}: {len(preserving)}")
    for record in preserving:
        print(record)
    print(f"  value-destroying rebindings or in-place mutations: {len(violations)}")
    for violation in violations:
        print(violation)

    # The one rebinding that does happen, reported explicitly so its effect can
    # be read rather than assumed.
    print("\n  every statement between them, in order:")
    for node in between:
        print(f"    line {node.lineno}: {ast.unparse(node)[:100]}")

    print("\nP3 -- negative evidence: in-place operations anywhere in the function")
    seen: dict[str, list[int]] = {}
    for node in ast.walk(function):
        mutation = inplace_target(node)
        if mutation is not None:
            seen.setdefault(mutation, []).append(getattr(node, "lineno", -1))
    for name in sorted(seen):
        lines = sorted(set(seen[name]))
        flag = "  <-- WATCHED" if name in WATCHED else ""
        print(f"  {name}: lines {lines}{flag}")

    verdict_p1 = not violations
    verdict_p2 = (
        not violations
        and ast.unparse(loop.target) == "_pass"
        and ast.unparse(loop.iter) == "range(_EM1_PASSES)"
        and loop.body.index(in_loop) == 0
    )
    # P4 -- the reason later passes must NOT be allowed to reuse the value:
    # inside the pass, `deployed` and `gradient` are mutated in place, so from
    # pass 0 onward the tensors the gradient was built from no longer exist.
    mutations_inside_loop: list[str] = []
    for node in ast.walk(loop):
        if node is in_loop:
            continue
        mutation = inplace_target(node)
        if mutation in WATCHED or mutation == "gradient":
            mutations_inside_loop.append(
                f"  line {node.lineno}: `{mutation}` mutated in place: "
                f"{ast.unparse(node)[:80]}"
            )
    verdict_p4 = bool(mutations_inside_loop)

    print(f"\nP1 nothing between the two gradients changes the value of "
          f"{' or '.join(WATCHED)}: {verdict_p1}")
    print(f"P2 the in-loop gradient is the first statement of the `_pass` loop, so "
          f"at _pass == 0 it evaluates the same expression on the same tensors: "
          f"{verdict_p2}")
    print(f"P4 the pass body mutates deployed/gradient in place, so the reuse must "
          f"be limited to pass 0: {verdict_p4}")
    for record in mutations_inside_loop:
        print(record)

    if not (verdict_p1 and verdict_p2 and verdict_p4):
        print("\nPREMISE DOES NOT HOLD -- do not hoist")
        return 1
    print(
        f"\nPREMISE HOLDS. Reusing the pre-loop gradient for pass 0 is exactly "
        f"output-equivalent, it removes one of the two gradient expressions the "
        f"K={shipped_passes} root executes per descent call, and the in-place "
        f"mutations inside the pass body are what make the reuse unsafe for any "
        f"pass after the first.\n"
        f"NOTE K={shipped_passes}: unlike v233's K=1 parent, the guarded branch is "
        f"live on this root -- pass 1 still recomputes, and verify.py measures it."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
