"""Compose A-GR1 onto the v245 root.

v236 (A-GR1 on v231) is a PURE APPEND: `diff -u v231 v236` is a single hunk
`@@ -12323,3 +12323,409 @@`, i.e. v231's last three lines (`flush=True,` / `)` /
`return corrected`) are followed by the A-GR1 block and nothing else in the file
changed.  The block only

  * defines `_AGR1_*` constants and helpers,
  * aliases `_AGR1_PARENT_CALIBRATION = hif4_calibration_attention`, then
  * rebinds `hif4_calibration_attention` to a wrapper that calls that alias.

So composing it onto any root that keeps the same module-level definition order
is mechanical: append the block.  The alias then captures the v245 wrapper
(which carries A-TF1/A-TG1), so A-GR1 trains on top of the optimised root.

WHAT THIS BUYS: A-GR1 is the largest Attention mechanism whose official side
isolation is POSITIVE (v234: +29 on the side-isolation channel, +50 on the R3
baseline) and whose full-package submissions v234/v236 timed out with 8 s and
9 s of headroom.  Its cost is calibration-side only.  The three v243-v245 cards
cut one attention calibration call by about 22% (GPU path) / 29% (CPU path), so
this composition puts that recovered margin against A-GR1's cost.

    .venv/Scripts/python.exe build_v246.py
"""

from __future__ import annotations

import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "solution.py"
BASE = ROOT / "solutions" / "20260911_v245_attention-atr1-rotation-seed-dedup_scoreNA_timeNA" / "solution.py"
DONOR = ROOT / "solutions" / "20260910_v236_attention-agr1-on-v231_scoreNA_timeNA" / "solution.py"
MARKER = b"# A-GR1 (2026-09-10): general asymmetric reciprocal matrix residual."


def agr1_block(donor: bytes) -> bytes:
    """The appended block: from the separator line that precedes the A-GR1 banner."""
    i = donor.index(MARKER)
    head = donor[:i]
    sep = head.rindex(b"# ---")
    return donor[sep:]


def main() -> int:
    base = BASE.read_bytes()
    donor = DONOR.read_bytes()
    block = agr1_block(donor)
    print(f"v245 base : {len(base)} B  sha256 {hashlib.sha256(base).hexdigest()[:16]}")
    print(f"A-GR1 block: {len(block)} B")
    if b"# A-GR1" not in block:
        raise SystemExit("block extraction failed")
    # v231 ends right after the L-EM2 activation wrapper; v245 ends after the
    # L-TF2/L-EM3 blocks instead.  The append point differs, but the EFFECT is
    # the same: the block lands after every definition, so its aliases
    # (_AGR1_PARENT_CALIBRATION / _AGR1_PARENT_Q/K/V) capture the same last
    # definitions in both files.  What matters is that no later definition
    # shadows them, which the AST check in the caller confirms.
    out = base.rstrip(b"\r\n") + b"\r\n\r\n\r\n" + block
    SRC.write_bytes(out)
    print(f"candidate : {len(out)} B  sha256 {hashlib.sha256(out).hexdigest().upper()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
