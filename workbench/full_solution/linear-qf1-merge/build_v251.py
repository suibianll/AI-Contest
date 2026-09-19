"""v251 = v250 (L-EM K=6 line) + L-QF1 local-quadratic correctness fix.

v232 measured the fix at +0.013739 on the Linear six-shard panel against the
then-current v230 root, but v232 was a sibling of v231: the fix never merged
into the v237 chain that v250 descends from, and both live shadows in v250
still carry the parent form.

The parent form

    quadratic = torch.einsum("krba,bij,krbj->krb", delta, local_gram, delta)

leaves `a` (first delta) and `i` (local_gram) as single-appearance indices, so
the computed quadratic is NOT delta^T G delta; for the pm1 single-element moves
the candidate set actually uses it reduces to a column/row sum of G instead of
G[m, m] (v232 report).  The fix renames the first delta index to `i`:

    quadratic = torch.einsum("krbi,bij,krbj->krb", delta, local_gram, delta)

which is the exact delta^T G delta the whole-row joint check already uses.

This is an append-only shadow: the last `_em1_dynamic_descent` definition (the
L-TF2 one) is copied to the end of the file with that single index renamed.
The live wrapper resolves the name at call time, so the appended copy wins.
"""

from __future__ import annotations

import hashlib
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
PARENT = ROOT / "solution.py"
OUT = HERE / "candidate_v251" / "solution.py"

EINSUM_PARENT = 'quadratic = torch.einsum("krba,bij,krbj->krb", delta, local_gram, delta)'
EINSUM_FIXED = 'quadratic = torch.einsum("krbi,bij,krbj->krb", delta, local_gram, delta)'
MARKER = "\n@torch.no_grad()\ndef _em1_dynamic_descent("

HEADER = '''

# ---------------------------------------------------------------------------
# L-QF1 on the v250 line: exact local quadratic in `_em1_dynamic_descent`.
#
# v232 derived this fix on the v230 (K = 1) root and measured +0.013739 on the
# Linear six-shard panel; it never merged into the v237 chain that this file
# descends from, so the parent form is still what runs here.
#
# The parent einsum `"krba,bij,krbj->krb"` leaves `a` (first delta) and `i`
# (local_gram) as single-appearance indices, and the expression it evaluates is
# not delta^T G delta.  For the eight pm1 moves this candidate set proposes,
# delta = c * e_m elementwise, so the true quadratic is c^2 * G[m, m] while the
# parent form reduces to c^2 times a column sum of G -- the acceptance test was
# scoring the wrong quantity.
#
# Renaming the first delta index to `i` gives the exact delta^T G delta that
# the whole-row joint check (`row_delta.mm(metric)`) already computes.  Only
# the acceptance test changes: the candidate set, the schedule, the coverage
# rule and the deployed output encoding are the parent's own code, and the
# append-only build never rewrites a parent byte.
#
# The definition below shadows the parent's (which stays in the file as dead
# code); the live wrapper resolves `_em1_dynamic_descent` at call time.
# ---------------------------------------------------------------------------


'''


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def main() -> int:
    parent_bytes = PARENT.read_bytes()
    parent_text = parent_bytes.decode("utf-8")

    count = parent_text.count(EINSUM_PARENT)
    assert count == 2, f"expected the parent einsum twice, found {count}"
    assert parent_text.count(EINSUM_FIXED) == 0, "fix already present"

    index = parent_text.rfind(MARKER)
    assert index >= 0, "last _em1_dynamic_descent not found"
    tail = parent_text[index:]
    assert tail.count(EINSUM_PARENT) == 1, "tail must carry exactly one cost line"
    assert tail.rstrip("\n").endswith(
        "return dict(result, mant=new_mant.reshape_as(mant), sign=new_sign.reshape_as(sign))"
    ), "tail is not the live shadow"

    fixed_tail = tail.replace(EINSUM_PARENT, EINSUM_FIXED)
    assert fixed_tail != tail
    assert fixed_tail.count(EINSUM_FIXED) == 1
    assert len(fixed_tail) == len(tail)

    header_bytes = HEADER.encode("utf-8")
    candidate_bytes = parent_bytes + header_bytes + fixed_tail.encode("utf-8")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_bytes(candidate_bytes)

    # --- verification -------------------------------------------------------
    assert candidate_bytes.startswith(parent_bytes), "not append-only"
    assert candidate_bytes.count(b"def _em1_dynamic_descent(") == (
        parent_bytes.count(b"def _em1_dynamic_descent(") + 1
    )
    appended = candidate_bytes[len(parent_bytes):].decode("utf-8")
    assert appended.count(EINSUM_FIXED) == 1
    assert appended.count(EINSUM_PARENT) == 0
    # the appended function body is the parent's tail with exactly one rename
    assert appended.startswith(HEADER)
    assert appended[len(HEADER):] == fixed_tail
    assert fixed_tail.replace(EINSUM_FIXED, EINSUM_PARENT) == tail

    size_parent = len(parent_bytes)
    size_candidate = len(candidate_bytes)
    print(f"parent  bytes={size_parent} sha={sha256(parent_bytes)[:16]}")
    print(f"candidate bytes={size_candidate} sha={sha256(candidate_bytes)[:16]}")
    print(f"appended bytes={size_candidate - size_parent} (header {len(header_bytes)} + tail {len(tail)})")
    print("APPEND-ONLY OK, single-index rename OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
