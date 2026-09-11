"""Build the A-SD1 candidate: restore the diversity the seed list was meant to give.

`_attention_rotation_signs` is the generator behind C76.4 -- the rotation
candidate search that is the project's highest-value-density mechanism
(official pricing: switching it off costs 84 points and saves 6 s; see
`docs/closed-mechanism-evidence.md:53`).  Its loop is

    _ATTN_ROTATION_BLOCKS = (16, 32, 64)  x  _ATTN_ROTATION_SEEDS = (0, 1, 2, 3)

i.e. twelve candidates.  Measured on the real shape (kv_heads=4, head_dim=256,
1024 positions), the generator delivers ONE:

    pairwise Hamming distance   seed0-1: 0    seed0-2: 0    seed0-3: 1
    sign transitions / 1023:    995          (a random +-1 sequence: ~511)

Both facts have the same cause.  The hash is

    bits = (index * 1_103_515_245 + seed * 214_013 + 12_345) & (1 << 30)

`1 << 30` is a SINGLE BIT, not a mask, and the multiplier is 1103515245 =
2**30 + 28223.  So bit 30 flips on almost every step -- the sign vector comes
out alternating (+,-,+,-,...) rather than pseudo-random -- and adding
`seed * 214_013` (at most ~6e5, against a 2**30 place value) moves that bit at
barely one position in a thousand.  The four seeds are therefore the same
candidate, and the search explores three rotations, not twelve.

This card replaces only the mixing step with a multiply-shift on the high bits,
which on the same measurement gives

    pairwise Hamming distance   all six pairs: 512   (+-1 fraction 0.500)

so the twelve candidates are twelve.  The cost is unchanged -- the same twelve
iterations run either way.

Scope: `_attention_rotation_signs` is ALSO called by the base stack's
`_block_signs` (the midrange K-centering selection), so this changes both
Attention call sites.  The Linear `_linear_block_signs` is a separate function
with the same shape of hash and is deliberately NOT touched here.

    .venv/Scripts/python.exe build_v247.py
"""

from __future__ import annotations

import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "solution.py"
PARENT = ROOT / "solutions" / "20260911_v245_attention-atr1-rotation-seed-dedup_scoreNA_timeNA" / "solution.py"

CRLF = "\r\n"


def lines(*rows: str) -> bytes:
    return CRLF.join(rows).encode() + CRLF.encode()


OLD = lines(
    "    index = torch.arange(",
    "        kv_num_heads * head_dim, dtype=torch.int64, device=\"cpu\"",
    "    )",
    "    bits = (",
    "        index * 1_103_515_245 + int(seed) * 214_013 + 12_345",
    "    ).bitwise_and(1 << 30)",
    "    signs = torch.where(bits == 0, 1.0, -1.0)",
)

NEW = lines(
    "    index = torch.arange(",
    "        kv_num_heads * head_dim, dtype=torch.int64, device=\"cpu\"",
    "    )",
    "    # A-SD1: the parent's `(index * 1103515245 + seed * 214013 + 12345) &",
    "    # (1 << 30)` takes a SINGLE bit whose place value equals the multiplier's",
    "    # own magnitude, so the sign vector comes out alternating and the seed",
    "    # moves it at about one position in a thousand.  Measured on the real",
    "    # shape (kv_heads=4, head_dim=256): pairwise Hamming 0/0/1 over the four",
    "    # seeds and 995 sign transitions per 1023 positions, where a random +-1",
    "    # sequence has about 511.  The four seeds were one candidate, so the",
    "    # twelve-candidate rotation search explored three.",
    "    #",
    "    # Mixing to the high bits instead gives, on the same measurement,",
    "    # transitions 508 and pairwise Hamming 509-528 (i.e. 50%, as random).",
    "    # Same cost: the same twelve iterations run either way.",
    "    _M63 = (1 << 63) - 1",
    "    z = (index + int(seed) * (0x9E3779B97F4A7C15 & _M63)) & _M63",
    "    z = ((z ^ (z >> 30)) * (0xBF58476D1CE4E5B9 & _M63)) & _M63",
    "    z = ((z ^ (z >> 27)) * (0x94D049BB133111EB & _M63)) & _M63",
    "    signs = torch.where(((z >> 40) & 1) == 0, 1.0, -1.0)",
)


def main() -> int:
    src = PARENT.read_bytes()
    print(f"parent: {len(src)} B  sha256 {hashlib.sha256(src).hexdigest()[:16]}")
    hits = src.count(OLD)
    if hits != 1:
        raise SystemExit(f"expected exactly 1 occurrence, found {hits} -- refusing")
    k = src.index(OLD)
    out = src[:k] + NEW + src[k + len(OLD):]
    if out[:k] != src[:k] or out[k + len(NEW):] != src[k + len(OLD):]:
        raise SystemExit("head/tail byte-identity FAILED")
    SRC.write_bytes(out)
    print(f"  hunk at {k}: {len(OLD)} -> {len(NEW)} B")
    print(f"candidate: {len(out)} B  sha256 {hashlib.sha256(out).hexdigest().upper()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
