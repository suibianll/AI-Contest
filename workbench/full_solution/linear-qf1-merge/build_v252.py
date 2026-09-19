"""v252 = v251 (v250 + L-QF1) + the A-GR1 block from v246.

v246's build is recorded as v245's bytes plus one appended A-GR1 block whose
aliases capture the v245 attention wrapper.  v251 keeps that wrapper
byte-identical (its two changes are Linear-side), so the same appended block
composes mechanically.  The block is taken from the archive as the bytes after
the v245/v246 common prefix; everything before it is untouched.
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
V251 = HERE / "candidate_v251" / "solution.py"
V245 = ROOT / "solutions/20260911_v245_attention-atr1-rotation-seed-dedup_scoreNA_timeNA/solution.py"
V246 = ROOT / "solutions/20260911_v246_attention-agr1-on-v245_scoreNA_timeNA/solution.py"
OUT = HERE / "candidate_v252" / "solution.py"


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def longest_common_prefix(left: bytes, right: bytes) -> int:
    limit = min(len(left), len(right))
    index = 0
    while index < limit and left[index] == right[index]:
        index += 1
    return index


def main() -> int:
    v251 = V251.read_bytes()
    v245 = V245.read_bytes()
    v246 = V246.read_bytes()

    boundary = longest_common_prefix(v245, v246)
    assert boundary == 519333, f"unexpected v245/v246 boundary {boundary}"
    block = v246[boundary:]
    assert len(block) == 15197, len(block)
    assert b"A-GR1 (2026-09-10)" in block
    assert b"_AGR1_PARENT_CALIBRATION" in block
    assert b"def hif4_calibration_attention(" in block

    candidate = v251 + block
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_bytes(candidate)

    assert candidate.startswith(v251), "not append-only vs v251"
    assert candidate.count(b"def _agr1_train(") == 1

    print(f"v251      bytes={len(v251)} sha={sha256(v251)[:16]}")
    print(f"block     bytes={len(block)}")
    print(f"v252      bytes={len(candidate)} sha={sha256(candidate)[:16]}")
    print("APPEND-ONLY OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
