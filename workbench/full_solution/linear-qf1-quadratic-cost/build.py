"""Build L-QF1 from the retained v230 Linear L-EM2 complete root.

L-QF1 corrects the local proposal cost in ``_em1_dynamic_descent``.  The parent
contracts

    quadratic = torch.einsum("krba,bij,krbj->krb", delta, local_gram, delta)

The summed index ``a`` of the first operand couples only to the first operand,
and the summed index ``i`` of the second only to the second, so the two sums do
not meet: the contraction evaluates to ``(sum_a delta_a) * sum_j
(sum_i G_ij) delta_j``.  The intended quantity is ``delta^T G delta``, which
needs the first operand to be indexed by the second operand's row index:

    quadratic = torch.einsum("krbi,bij,krbj->krb", delta, local_gram, delta)

Nothing else changes.  K, the eight pm1 candidates, the 16-step group-major
schedule, the coverage rule, the scale/hierarchy/sign semantics, the linear
term and the whole-row joint acceptance are the parent's own code.

The parent's own weight encoder already contracts the same way it should
(``solution.py:1983`` uses ``"krbi,bij,krbj->krb"`` for ``e^T H e``), so the
corrected form is the file's own idiom rather than a new one.

Build strategy -- the appended module is *derived from the parent*, not written
by hand:

  1. cut the parent's ``_em1_dynamic_descent`` source out of the parent bytes;
  2. substitute exactly one substring, ``krba,bij,krbj->krb`` ->
     ``krbi,bij,krbj->krb``;
  3. append that text (with a comment header) after a fixed separator.

Because the appended code is the parent's own text, the "sole change" claim is
mechanically checkable rather than asserted: this script reports the number of
differing bytes between the parent's function text and the appended one, and
refuses to build unless it is exactly 1.

``_em1_dynamic_descent`` is resolved from module globals when the parent's
``hif4_dynamic_quantize_activation`` runs, so the appended definition shadows
the parent's without that hook being touched at all.  Both hook names, the
Attention APIs, the shared helpers and every archived file stay byte-identical;
the only dead code in the candidate is the now-unreachable parent definition.
"""

from pathlib import Path
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

BUGGY = "krba,bij,krbj->krb"
CORRECT = "krbi,bij,krbj->krb"

HEADER = '''# ---------------------------------------------------------------------------
# L-QF1 -- local proposal quadratic of _em1_dynamic_descent, corrected.
#
# One edit: the einsum contraction of the local cost.
#
#   parent   quadratic = torch.einsum("krba,bij,krbj->krb", delta, local_gram, delta)
#   L-QF1    quadratic = torch.einsum("krbi,bij,krbj->krb", delta, local_gram, delta)
#
# In the parent form the first operand's summed index `a` couples only to the
# first operand and the second operand's summed index `i` only to the second,
# so the contraction is (sum_a delta_a) * sum_j (sum_i G_ij) delta_j -- the
# column sums of G against the total displacement, not delta^T G delta.  The
# corrected form couples `i` across both operands and evaluates delta^T G delta.
#
# `krbi,bij,krbj->krb` is the parent file's own idiom for a quadratic form: its
# weight encoder contracts e^T H e exactly this way at solution.py:1983.
#
# Sole change.  K, the eight pm1 candidates, the 16-step group-major schedule,
# the coverage rule, the scale/hierarchy/sign semantics, the linear term and the
# whole-row joint acceptance are the parent's own code, byte for byte.  This
# module is the parent's own _em1_dynamic_descent text with that one subscript
# string substituted; build.py reports the differing-byte count (1).
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


def build() -> dict:
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

    if parent_fn_text.count(BUGGY) != 1:
        raise SystemExit(
            f"parent function holds {parent_fn_text.count(BUGGY)} copies of the "
            "buggy contraction, expected exactly 1"
        )
    if CORRECT in parent_fn_text:
        raise SystemExit("parent function already holds the corrected contraction")

    fixed_fn_text = parent_fn_text.replace(BUGGY, CORRECT)

    # The sole-change guarantee, measured on the bytes themselves.
    a = parent_fn_text.encode("utf-8")
    b = fixed_fn_text.encode("utf-8")
    if len(a) != len(b):
        raise SystemExit(f"edit changed the byte length: {len(a)} -> {len(b)}")
    differing = sum(1 for x, y in zip(a, b) if x != y)
    if differing != 1:
        raise SystemExit(f"edit touches {differing} bytes, expected exactly 1")

    module_text = HEADER + fixed_fn_text.rstrip("\n") + "\n"
    module_bytes = module_text.encode("utf-8")

    # The header quotes both spellings in order to document the edit, so counts
    # over the whole module would depend on the header's prose.  Take the module
    # apart instead: everything after the header must be the function, and the
    # function must hold the corrected string exactly once and the buggy one
    # never.  That is the guarantee that matters.
    fn_part = fixed_fn_text.rstrip("\n") + "\n"
    if module_text != HEADER + fn_part:
        raise SystemExit("generated module is not the header plus the function")
    if fn_part.count(BUGGY) != 0:
        raise SystemExit("generated function still holds the buggy contraction")
    if fn_part.count(CORRECT) != 1:
        raise SystemExit(
            f"generated function holds {fn_part.count(CORRECT)} corrected "
            "contractions, expected exactly 1"
        )

    candidate_bytes = (
        parent_bytes.rstrip(b"\n") + b"\n\n\n" + module_bytes.rstrip(b"\n") + b"\n"
    )

    # The candidate must re-bind the descent, and the parent's call site must
    # be the only caller, so the shadowing definition is the live one.
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
        "extracted_function_differing_bytes": differing,
        "edit": f"{BUGGY} -> {CORRECT}",
    }
    BUILD_JSON.write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")
    return info


if __name__ == "__main__":
    for key, value in build().items():
        print(f"{key}={value}")
