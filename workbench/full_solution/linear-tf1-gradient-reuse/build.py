"""Build L-TF1 from the retained v230 Linear L-EM2 complete root.

L-TF1 removes one redundant gradient evaluation.  At ``_EM1_PASSES = 1`` the
parent computes

    gradient = (deployed - reference).mm(metric) + reference.mm(h_matrix)

twice per call: once before the pass loop, where it is checked for finiteness,
and then again as the first statement of the single pass, where the pre-loop
result is rebound and discarded.  The two evaluations are the same expression
on the same tensors -- ``premise.py`` confirms that statically and ``verify.py``
confirms it at runtime bit for bit -- so the first result can simply be reused
and the second evaluation dropped.

The edit guards the in-loop recomputation with the pass counter:

    for _pass in range(_EM1_PASSES):
        if _pass:
            gradient = (deployed - reference).mm(metric) + reference.mm(h_matrix)
            if not torch.isfinite(gradient).all():
                aborted = True
                break

At ``_pass == 0`` the pre-loop tensor is used, and nothing else changes.  At
``_pass > 0`` the recomputation still happens, in the same order, because the
pass body mutates ``deployed`` and ``gradient`` in place -- ``deployed.add_``
and ``gradient.add_`` -- so from the first pass onward the tensors the gradient
was built from no longer exist.  That is why the guard is ``if _pass`` and not
a hoist of the whole expression out of the loop.

Build strategy -- the appended module is *derived from the parent*, not written
by hand:

  1. cut the parent's ``_em1_dynamic_descent`` source out of the parent bytes;
  2. substitute exactly one substring, the unguarded loop head, for the guarded
     one;
  3. append that text (with a comment header) after a fixed separator.

The sole-change claim is then checked two independent ways, both computed rather
than asserted:

  * byte level -- the edit is one substring substitution, and the script reports
    the net byte delta and the changed-line count and refuses to build if the
    needle is not unique or the replacement is already present;
  * syntax level -- the candidate function must parse to an AST that is
    *identical* to the parent's once the one inserted ``if _pass:`` node is
    dissolved back into its parent block.  That is a stronger statement than a
    byte count: it says the executed statement tree is unchanged apart from the
    guard.

``_em1_dynamic_descent`` is resolved from module globals when the parent's
``hif4_dynamic_quantize_activation`` runs, so the appended definition shadows
the parent's without that hook being touched at all.  Both hook names, the
Attention APIs, the shared helpers and every archived file stay byte-identical;
the only dead code in the candidate is the now-unreachable parent definition.
"""

from pathlib import Path
import ast
import difflib
import hashlib
import json


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
PARENT = ROOT / "solution.py"
CANDIDATE = HERE / "candidate" / "solution.py"
MODULE_OUT = HERE / "implementation.generated.py"
BUILD_JSON = HERE / "build.json"

EXPECTED_PARENT_SHA256 = (
    "0f1af6dbc207ff32b2c6be16987e9c4fe50f3f10747de26782ef52a6f2fab7bc"
)
EXPECTED_PARENT_BYTES = 505496

FN_START = b"@torch.no_grad()\ndef _em1_dynamic_descent(\n"
FN_END = b"\n_EM1_PARENT_LINEAR_CALIBRATION = hif4_calibration_and_quantize_weight"

OLD_HEAD = """    for _pass in range(_EM1_PASSES):
        gradient = (deployed - reference).mm(metric) + reference.mm(h_matrix)
        if not torch.isfinite(gradient).all():
            aborted = True
            break
"""

NEW_HEAD = """    for _pass in range(_EM1_PASSES):
        if _pass:
            gradient = (deployed - reference).mm(metric) + reference.mm(h_matrix)
            if not torch.isfinite(gradient).all():
                aborted = True
                break
"""

HEADER = '''# ---------------------------------------------------------------------------
# L-TF1 -- the first pass reuses the gradient the parent already computed.
#
# At _EM1_PASSES = 1 the parent evaluates
#
#     gradient = (deployed - reference).mm(metric) + reference.mm(h_matrix)
#
# twice per call: once before the pass loop, where the result is checked for
# finiteness, and again as the first statement of the single pass, which rebinds
# the name and discards the earlier tensor.  The two evaluations are the same
# expression on the same tensors -- premise.py confirms statically that nothing
# between them changes `deployed` or `reference`, and verify.py confirms at
# runtime that the two results are bit-identical -- so the first one is reused
# and the second is not performed:
#
#   parent   for _pass in range(_EM1_PASSES):
#                gradient = (deployed - reference).mm(metric) + reference.mm(h_matrix)
#                if not torch.isfinite(gradient).all():
#                    aborted = True
#                    break
#
#   L-TF1    for _pass in range(_EM1_PASSES):
#                if _pass:
#                    gradient = (deployed - reference).mm(metric) + reference.mm(h_matrix)
#                    if not torch.isfinite(gradient).all():
#                        aborted = True
#                        break
#
# The guard stops the reuse at pass 0 for a reason the code itself gives: the
# pass body mutates `deployed` and `gradient` in place (deployed.add_ and
# gradient.add_), so from the first pass onward the tensors the gradient was
# built from no longer exist and every later pass must still recompute, in the
# parent's own order.  At K = 1, which is the shipped root, the guarded branch
# never runs.
#
# Sole change.  The pre-loop finiteness check and its `nonfinite-gradient`
# diagnostic, the metric and its ridge, the candidate set, the 16-step
# group-major schedule, the coverage rule, the local cost and the whole-row
# joint acceptance are the parent's own code, byte for byte.  build.py reports
# both the byte delta and an AST comparison: the candidate's function
# parses to the parent's statement tree exactly, once the inserted `if _pass:`
# node is dissolved back into the pass body.
#
# The definition below shadows the parent's, which stays in the file as dead
# code -- the append-only build never rewrites a parent byte.
# ---------------------------------------------------------------------------


'''


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def extract_parent_function(parent_bytes: bytes) -> bytes:
    if parent_bytes.count(FN_START) != 1:
        raise SystemExit("parent function start anchor is not unique")
    if parent_bytes.count(FN_END) != 1:
        raise SystemExit("parent function end anchor is not unique")
    start = parent_bytes.index(FN_START)
    end = parent_bytes.index(FN_END, start)
    if end <= start:
        raise SystemExit("parent function end precedes its start")
    return parent_bytes[start:end]


def dissolve_pass_guard(tree: ast.Module) -> ast.Module:
    """Removes the inserted ``if _pass:`` node by splicing its body back in.

    Used only as a comparison device: if dissolving the guard makes the
    candidate's AST equal the parent's, then the guard is the only structural
    difference between them.  Mutates and returns the tree.
    """
    function = tree.body[0]
    guard = None
    for node in ast.walk(function):
        if isinstance(node, ast.For) and ast.unparse(node.target) == "_pass":
            first = node.body[0]
            if (
                isinstance(first, ast.If)
                and ast.unparse(first.test) == "_pass"
                and first.orelse == []
            ):
                guard = (node, first)
                break
    if guard is None:
        raise SystemExit("candidate does not carry the expected `if _pass:` guard")
    loop, node = guard
    index = loop.body.index(node)
    loop.body[index : index + 1] = node.body
    ast.fix_missing_locations(tree)
    return tree


def main() -> dict:
    parent_bytes = PARENT.read_bytes()
    parent_digest = sha256_bytes(parent_bytes)
    if parent_digest != EXPECTED_PARENT_SHA256:
        raise SystemExit(
            f"parent SHA256 mismatch: {parent_digest} != {EXPECTED_PARENT_SHA256}"
        )
    if len(parent_bytes) != EXPECTED_PARENT_BYTES:
        raise SystemExit(
            f"parent byte count mismatch: {len(parent_bytes)} != {EXPECTED_PARENT_BYTES}"
        )

    parent_fn = extract_parent_function(parent_bytes)
    parent_fn_text = parent_fn.decode("utf-8")

    if parent_fn_text.count(OLD_HEAD) != 1:
        raise SystemExit(
            f"parent function holds {parent_fn_text.count(OLD_HEAD)} copies of the "
            "unguarded loop head, expected exactly 1"
        )
    if NEW_HEAD in parent_fn_text:
        raise SystemExit("parent function already holds the guarded loop head")

    fixed_fn_text = parent_fn_text.replace(OLD_HEAD, NEW_HEAD)

    # Byte-level sole-change evidence.  A positional byte-by-byte count would be
    # meaningless here -- the edit inserts a line, so everything after it shifts
    # and a positional comparison would report the whole tail as differing.  The
    # edit is exactly one substring substitution; what a reader wants to know is
    # how large it is, so report the byte delta and a real line diff.
    a = parent_fn_text.encode("utf-8")
    b = fixed_fn_text.encode("utf-8")
    line_diff = [
        opcode
        for opcode in difflib.SequenceMatcher(
            None, parent_fn_text.splitlines(), fixed_fn_text.splitlines()
        ).get_opcodes()
        if opcode[0] != "equal"
    ]
    changed_lines = sum(max(o[2] - o[1], o[4] - o[3]) for o in line_diff)

    # Syntax-level sole-change evidence: dissolving the guard must reproduce the
    # parent's statement tree exactly.
    parent_tree = ast.parse(parent_fn_text)
    candidate_tree = ast.parse(fixed_fn_text)
    if ast.dump(parent_tree) != ast.dump(dissolve_pass_guard(candidate_tree)):
        raise SystemExit(
            "the candidate's AST is not the parent's once the `if _pass:` guard "
            "is dissolved -- the edit is not the sole structural change"
        )
    tree_identical_after_dissolving_guard = True

    module_text = HEADER + fixed_fn_text.rstrip("\n") + "\n"
    module_bytes = module_text.encode("utf-8")

    # The header quotes both spellings in order to document the edit, so counts
    # over the whole module would depend on the header's prose.  Take the module
    # apart instead: everything after the header must be the function.
    fn_part = fixed_fn_text.rstrip("\n") + "\n"
    if module_text != HEADER + fn_part:
        raise SystemExit("generated module is not the header plus the function")
    if fn_part.count("if _pass:") != 1:
        raise SystemExit("generated function should hold exactly one `if _pass:` guard")

    candidate_bytes = (
        parent_bytes.rstrip(b"\n") + b"\n\n\n" + module_bytes.rstrip(b"\n") + b"\n"
    )

    if candidate_bytes.count(b"def _em1_dynamic_descent(") != 2:
        raise SystemExit("candidate should hold the parent definition plus the shadow")
    if candidate_bytes.count(b"corrected = _em1_dynamic_descent(") != 1:
        raise SystemExit("candidate should hold exactly one call site")
    if not candidate_bytes.startswith(parent_bytes.rstrip(b"\n")):
        raise SystemExit("candidate is not an append of the parent bytes")

    CANDIDATE.parent.mkdir(parents=True, exist_ok=True)
    CANDIDATE.write_bytes(candidate_bytes)
    MODULE_OUT.write_bytes(module_bytes)

    info = {
        "parent_path": str(PARENT.relative_to(ROOT)).replace("\\", "/"),
        "parent_sha256": parent_digest,
        "parent_bytes": len(parent_bytes),
        "implementation_sha256": sha256_bytes(module_bytes),
        "implementation_bytes": len(module_bytes),
        "candidate_sha256": sha256_bytes(candidate_bytes),
        "candidate_bytes": len(candidate_bytes),
        "extracted_parent_function_bytes": len(a),
        "substitutions_performed": 1,
        "extracted_function_changed_lines": changed_lines,
        "extracted_function_byte_delta": len(b) - len(a),
        "ast_identical_after_dissolving_pass_guard": tree_identical_after_dissolving_guard,
        "edit": "guard the in-loop gradient recomputation with `if _pass:`",
    }
    BUILD_JSON.write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")
    return info


if __name__ == "__main__":
    for key, value in main().items():
        print(f"{key}={value}")
