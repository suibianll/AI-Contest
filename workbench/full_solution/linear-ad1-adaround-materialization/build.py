"""Builds the L-AD1 candidate: the 16-pattern enumeration without the int64
materialization.

Plan section 4.3 fixes the card.  ``_adaround_mantissa`` currently computes the
two integral floor/ceil codes, then builds the 16 four-bit patterns with
``torch.where`` over an int64 tensor that is 16x the payload, casts the whole
16x tensor to float32, and scales it by 0.25.  The measured cost of this
function is 51-68% of the Gram-layer encoder, and the encoder is op-count bound
(425-1068 aten ops per call, identical across a 4x change in rows), so the
16x-wide int64 tensor and its conversion are exactly the kind of traffic the
card is allowed to remove.

What is removed is the materialization, not the enumeration: the 16 candidate
patterns are still all built and still all scored.  The codes stay int64 on the
small ``[..., 4]`` tensor -- the cast and the 0.25 scale are exact for integer
codes -- so both are computed before the ``where`` instead of after it.  The
selection, the loss, the argmin, the gather, the coverage and the refinement
rounds are untouched; the changed statements are a window in the middle of the
function and the statements after it are AST-identical.

``_adaround_mantissa`` is a shared helper: the weight/static paths call it too
(``_solve_exact_hierarchy`` at 3513, ``_dense_to_hif4`` at 3780, and
``_solve_exact_hierarchy`` itself has callers at 1475/1970/7222/7258/7427/7463).
The equivalence obligation is therefore for every caller, and it is discharged
in ``verify.py`` by bit comparison, not by tolerance.

The candidate is the parent's bytes with a module appended, so the parent is
untouched: the appended shadow definition is resolved from the module globals at
call time and wins over the earlier one, without a hook and without editing a
single parent byte.
"""

from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
PARENT = ROOT / "solution.py"

# The promoted root is v231, not the v230 root this card was first drafted
# against: v231 scored 18518/291 s officially (v230: 18428/292 s) and the working
# tree was promoted to it, so a candidate built on v230 would silently revert the
# K = 2 arm that the official run just paid for.  The drift is caught rather than
# absorbed because the sha below is checked before a byte is written.
EXPECTED_PARENT_SHA256 = (
    "ea79a1c12dc667142c620975aab188920fae7b29988c804f41c1a696cc5754f1"
)
EXPECTED_PARENT_BYTES = 505762
PARENT_ROLE = (
    "v231 archived root: Linear L-EM3 groupstep K=2 + v195 Attention, "
    "official 18518 / 291 s"
)
# The superseded root, kept only so the rebase is legible in build.json.
DECLARED_PREVIOUS_PARENT_SHA256 = (
    "0f1af6dbc207ff32b2c6be16987e9c4fe50f3f10747de26782ef52a6f2fab7bc"
)

FUNCTION = "_adaround_mantissa"
HELPER = "_AD1_pattern_masks"

# The two anchor lines bracket the block that is reorganized.  They are located
# in the parent's own source rather than transcribed, so the needle below cannot
# drift from the parent by a whitespace character.
BLOCK_FIRST = "    raw_code = x_abs * (4.0 / local_scale)"
BLOCK_LAST = "    all_mantissa = all_codes.to(torch.float32) * 0.25"

REPLACEMENT = '''\
    raw_code = x_abs * (4.0 / local_scale)          # continuous code
    floor_code = torch.floor(raw_code).clamp(0, 6).to(torch.int64)
    ceil_code = torch.ceil(raw_code).clamp(0, 7).to(torch.int64)
    # Scale the two integral codes before the 16-pattern expansion.  A code is
    # an integer in [0, 7] and 0.25 is a power of two, so `code.to(torch.float32)
    # * 0.25` is exact; doing it on the small [..., 4] tensor instead of on the
    # 16x tensor leaves the candidate mantissas value for value unchanged, while
    # the 16x `where` no longer has to carry int64 and the 16x tensor no longer
    # needs a second conversion pass.
    floor_mant = floor_code.to(torch.float32) * 0.25
    ceil_mant = ceil_code.to(torch.float32) * 0.25

    # The 16 bit masks depend on nothing but the device, so they are built once
    # and reused.  They are read-only here: `.reshape` below returns a view and
    # nothing writes through it.
    masks = _AD1_pattern_masks(x_abs.device)
    # masks: [16, 4]

    # Build all 16 candidate mantissas: [16, ..., 4]
    # Expand masks to match x_abs shape
    prefix_ndim = x_abs.ndim - 1  # all dims except last (size 4)
    # masks needs shape [16, 1, 1, ..., 1, 4] for broadcasting
    mask_expanded = masks.reshape(16, *([1] * prefix_ndim), 4)
    all_mantissa = torch.where(
        mask_expanded, ceil_mant.unsqueeze(0), floor_mant.unsqueeze(0)
    )
'''

APPENDED = '''\
# ---------------------------------------------------------------------------
# L-AD1: the 16-pattern AdaRound enumeration without the int64 materialization.
#
# `_adaround_mantissa` above is left byte for byte as it was; the definition
# below shadows it from this point on, because the module-global lookup that
# resolves it happens at call time.  Nothing else in this file is redefined.
#
# The enumeration is unchanged: all 16 four-bit patterns are still built and
# still scored by the same batched einsum.  Only the order of two operations
# inside that construction changed -- the code-to-mantissa conversion now
# happens on the small [..., 4] tensors rather than on the 16x tensor, and the
# 16x `where` now produces the mantissas directly.  Both conversions are exact
# for integer codes, so every candidate is bit-identical.
# ---------------------------------------------------------------------------


_AD1_PATTERN_MASK_CACHE: dict = {}


def _AD1_pattern_masks(device: torch.device) -> torch.Tensor:
    """The 16 four-bit patterns as a ``[16, 4]`` bool tensor, one per device.

    The values depend on nothing but the device, so the tensor is built once per
    device and handed out to every call.  It is only ever read: the caller
    reshapes it (a view, no copy) and passes it to ``torch.where`` as an input.
    """

    key = str(torch.device(device))
    masks = _AD1_PATTERN_MASK_CACHE.get(key)
    if masks is None:
        bits_all = torch.arange(16, device=device)
        masks = (
            (bits_all.unsqueeze(1) >> torch.arange(4, device=device)) & 1
        ).bool()
        _AD1_PATTERN_MASK_CACHE[key] = masks
    return masks
'''


def function_source(tree: ast.Module, name: str, lines: list[str]) -> str:
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return "\n".join(lines[node.lineno - 1 : node.end_lineno])
    raise SystemExit(f"module has no top-level function {name}")


def statements(tree: ast.Module) -> list[str]:
    return [ast.dump(node) for node in tree.body]


def main() -> int:
    parent_bytes = PARENT.read_bytes()
    digest = hashlib.sha256(parent_bytes).hexdigest()
    if digest != EXPECTED_PARENT_SHA256:
        raise SystemExit(f"parent sha256 {digest} != {EXPECTED_PARENT_SHA256}")
    if len(parent_bytes) != EXPECTED_PARENT_BYTES:
        raise SystemExit(f"parent is {len(parent_bytes)} bytes, expected "
                         f"{EXPECTED_PARENT_BYTES}")

    text = parent_bytes.decode("utf-8")
    lines = text.split("\n")
    parent_tree = ast.parse(text)
    source = function_source(parent_tree, FUNCTION, lines)

    # Locate the block from the parent's own text, by content.
    fn_lines = source.split("\n")
    start = next(
        (i for i, line in enumerate(fn_lines) if line.startswith(BLOCK_FIRST)), None
    )
    end = next(
        (i for i, line in enumerate(fn_lines) if line.startswith(BLOCK_LAST)), None
    )
    if start is None or end is None or end < start:
        raise SystemExit("could not locate the block in the parent function")
    needle = "\n".join(fn_lines[start : end + 1])
    if source.count(needle) != 1:
        raise SystemExit(f"block occurs {source.count(needle)} times, expected 1")

    candidate_source = source.replace(needle, REPLACEMENT.rstrip("\n"))
    if candidate_source == source:
        raise SystemExit("substitution changed nothing")

    # The two names the block introduced that the rewrite drops must be gone
    # from the function entirely, and the two it introduces must be bound once
    # and read once.  (That the tail does not read `floor_code`/`ceil_code`/
    # `all_codes` is proved below by the AST suffix comparison.)
    for dropped in ("bits_all", "all_codes"):
        occurrences = candidate_source.count(dropped)
        if occurrences != 0:
            raise SystemExit(
                f"{dropped} still appears {occurrences} times in the candidate"
            )
    for introduced in ("floor_mant", "ceil_mant"):
        occurrences = candidate_source.count(introduced)
        if occurrences != 2:
            raise SystemExit(
                f"{introduced} appears {occurrences} times in the candidate, "
                f"expected 2 (one binding, one read)"
            )

    head = parent_bytes.rstrip(b"\n")
    module_bytes = (APPENDED.rstrip("\n") + "\n\n\n" + candidate_source.rstrip("\n") + "\n")
    candidate_bytes = head + b"\n\n\n" + module_bytes.encode("utf-8")
    if not candidate_bytes.startswith(head):
        raise SystemExit("candidate does not begin with the parent")
    candidate_text = candidate_bytes.decode("utf-8")
    candidate_tree = ast.parse(candidate_text)

    # Sole change, computed rather than asserted: the candidate's first
    # statements are AST-identical to the parent's, and exactly two statements
    # follow them.
    parent_top = statements(parent_tree)
    candidate_top = statements(candidate_tree)
    added = len(candidate_top) - len(parent_top)
    if added != 3:
        raise SystemExit(f"{added} statements appended, expected 3")
    if candidate_top[: len(parent_top)] != parent_top:
        raise SystemExit("the parent's own statements are not unchanged")
    appended_kinds = [
        type(node).__name__ for node in candidate_tree.body[len(parent_top) :]
    ]
    if appended_kinds != ["AnnAssign", "FunctionDef", "FunctionDef"]:
        raise SystemExit(f"appended statement kinds are {appended_kinds}")

    appended_nodes = [
        node
        for node in candidate_tree.body[len(parent_top) :]
        if isinstance(node, ast.FunctionDef)
    ]
    appended_names = [node.name for node in appended_nodes]
    if appended_names != [HELPER, FUNCTION]:
        raise SystemExit(f"appended definitions are {appended_names}")

    shadow = appended_nodes[1]
    parent_fn = next(
        node
        for node in parent_tree.body
        if isinstance(node, ast.FunctionDef) and node.name == FUNCTION
    )
    parent_body = [ast.dump(node) for node in parent_fn.body]
    shadow_body = [ast.dump(node) for node in shadow.body]
    if len(parent_body) != len(shadow_body):
        raise SystemExit(
            f"statement count changed: {len(parent_body)} -> {len(shadow_body)}"
        )
    prefix = 0
    while (
        prefix < len(parent_body) and parent_body[prefix] == shadow_body[prefix]
    ):
        prefix += 1
    suffix = 0
    while (
        suffix < len(parent_body) - prefix
        and parent_body[-1 - suffix] == shadow_body[-1 - suffix]
    ):
        suffix += 1
    changed = [i for i in range(len(parent_body)) if parent_body[i] != shadow_body[i]]
    expected_changed = list(range(prefix, len(parent_body) - suffix))
    if changed != expected_changed:
        raise SystemExit("the changed statements are not one contiguous window")

    out = HERE / "implementation.generated.py"
    out.write_bytes(candidate_bytes)

    report = {
        "parent": str(PARENT),
        "parent_role": PARENT_ROLE,
        "parent_sha256": digest,
        "parent_bytes": len(parent_bytes),
        "previous_parent_sha256": DECLARED_PREVIOUS_PARENT_SHA256,
        "rebased_from_previous_parent": True,
        "rebase_reason": (
            "v231 was promoted to the working tree after this card was drafted; "
            "building on the older v230 root would revert the L-EM3 K = 2 arm "
            "that the official run scored at 18518 / 291 s"
        ),
        "candidate": str(out),
        "candidate_sha256": hashlib.sha256(candidate_bytes).hexdigest(),
        "candidate_bytes": len(candidate_bytes),
        "appended_bytes": len(candidate_bytes) - len(head),
        "function": FUNCTION,
        "helper": HELPER,
        "statements_in_parent_module": len(parent_top),
        "statements_appended": added,
        "appended_definitions": appended_names,
        "parent_module_prefix_ast_identical": True,
        "function_statement_count": len(parent_body),
        "function_prefix_identical": prefix,
        "function_suffix_identical": suffix,
        "function_changed_statement_indices": changed,
        "substitutions_performed": 1,
        "block_bytes_replaced": len(needle),
        "block_bytes_written": len(REPLACEMENT.rstrip("\n")),
    }
    (HERE / "build.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )

    print(f"parent   {report['parent_sha256']}  {report['parent_bytes']} bytes")
    print(f"candidate {report['candidate_sha256']}  {report['candidate_bytes']} bytes"
          f"  (+{report['appended_bytes']})")
    print(f"substitutions performed: {report['substitutions_performed']}")
    print(f"parent module statements: {len(parent_top)}  appended: {added} "
          f"({', '.join(appended_names)})")
    print(f"parent module prefix AST-identical: yes")
    print(f"function statements: {len(parent_body)}  "
          f"prefix identical: {prefix}  suffix identical: {suffix}  "
          f"changed: {changed[0]}..{changed[-1]}")
    print(f"block replaced: {len(needle)} bytes -> "
          f"{len(REPLACEMENT.rstrip(chr(10)))} bytes")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
