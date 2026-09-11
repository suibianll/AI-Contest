"""Build the A-TR1 candidate by byte-level replacement of the v244 root.

One contiguous replacement.  Uniqueness assert + head/tail byte-identity per
hunk, and a segment comparison across the hunk, because a `cmp` prefix match
does NOT prove a single hunk (defect #37).

    .venv/Scripts/python.exe build_v245.py
"""

from __future__ import annotations

import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "solution.py"
PARENT = ROOT / "solutions" / "20260911_v244_attention-atg1-gate-standard-share_scoreNA_timeNA" / "solution.py"

CRLF = "\r\n"


def lines(*rows: str) -> bytes:
    return CRLF.join(rows).encode() + CRLF.encode()


OLD = lines(
    "            for seed in _ATTN_ROTATION_SEEDS:",
    "                signs = _attention_rotation_signs(kv_num_heads, head_dim, int(seed))",
    "                rotation_q_state, rotation_k_state = _build_qk_states(",
)

NEW = lines(
    "            # A-TR1: _attention_rotation_signs depends ONLY on the seed, but the",
    "            # loop is over (block_size, seed).  At this shape seeds 0, 1 and 2",
    "            # produce the identical sign vector and only seed 3 differs, so each",
    "            # block re-scores the same candidate three times.  The candidate state",
    "            # is a function of (signs, block) and not of the seed, and the",
    "            # selection below is a strict `<` on a score that is therefore also",
    "            # identical, so a duplicate iteration can never change the argmax.",
    "            # Skipping duplicates leaves every result bit-identical.",
    "            distinct_signs: list = []",
    "            for seed in _ATTN_ROTATION_SEEDS:",
    "                signs = _attention_rotation_signs(kv_num_heads, head_dim, int(seed))",
    "                if any(torch.equal(signs, seen) for seen in distinct_signs):",
    "                    continue",
    "                distinct_signs.append(signs)",
    "                rotation_q_state, rotation_k_state = _build_qk_states(",
)


def main() -> int:
    src = PARENT.read_bytes()
    print(f"parent: {len(src)} B  sha256 {hashlib.sha256(src).hexdigest().upper()}")
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
