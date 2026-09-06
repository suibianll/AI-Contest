"""Assemble the L2 candidate (v160 stack + rank-2 residual, standard attention).

Mechanical transplant only: the candidate is the committed v163 side package
(v160 Linear + appended standard Attention) plus exactly the five Linear-side
hunks of the committed v163->v182 diff (rank-2 residual redistribution).  No
new algorithm code is introduced; every byte comes from the two archived,
previously reviewed sources.  Attention-side hunks and the appended-standard-
block removal hunk are deliberately NOT applied, so the frozen Attention side
stays the v163/v162 standard block.

Self-verification before writing:
1. applying the FULL diff to v163 must reproduce v182 byte-exactly (proves the
   applier and the diff input);
2. the assembled file must compile and expose the six API names;
3. SHA256 of both sources and the output is printed for the manifest.

Usage:
  .venv/Scripts/python.exe workbench/v162_linear/build_l2_candidate.py
"""

from __future__ import annotations

import hashlib
import py_compile
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
V163 = ROOT / "solutions" / "20260903_v163_v160-linear_standard-attn_scoreNA_timeNA" / "solution.py"
V182 = ROOT / "solutions" / "20260904_v182_rank2-linear_v180-attn_scoreNA_timeNA" / "solution.py"
OUT = ROOT / "workbench" / "v162_linear" / "candidate" / "solution.py"
DIFF = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/tmp/v163_v182.diff")

APPLY_OLD_STARTS = {71, 8155, 8218, 8429, 8473}
SKIP_OLD_STARTS = {365, 8761, 10188, 10295}

API_NAMES = [
    "hif4_calibration_and_quantize_weight",
    "hif4_dynamic_quantize_activation",
    "hif4_calibration_attention",
    "hif4_dynamic_quantize_q",
    "hif4_dynamic_quantize_k",
    "hif4_dynamic_quantize_v",
]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def parse_hunks(diff_text: str) -> list[dict]:
    hunks: list[dict] = []
    current: dict | None = None
    for line in diff_text.splitlines(keepends=True):
        if line.startswith("@@"):
            m = re.match(r"@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@", line)
            assert m, f"bad hunk header: {line!r}"
            current = {
                "old_start": int(m.group(1)),
                "lines": [],
            }
            hunks.append(current)
        elif line.startswith(("diff ", "index ", "--- ", "+++ ", "warning:")):
            current = None
        elif current is not None:
            if line.startswith("\\"):  # "\ No newline at end of file"
                continue
            assert line[:1] in (" ", "+", "-"), f"unexpected diff line: {line!r}"
            current["lines"].append(line)
    return hunks


def apply_hunks(src: str, hunks: list[dict], selected: set[int]) -> str:
    src_lines = src.splitlines(keepends=True)
    out: list[str] = []
    src_line = 1
    for hunk in hunks:
        old_start = hunk["old_start"]
        while src_line < old_start:
            out.append(src_lines[src_line - 1])
            src_line += 1
        for line in hunk["lines"]:
            tag, body = line[:1], line[1:]
            if tag == " ":
                assert src_lines[src_line - 1] == body or body == "", (
                    f"context mismatch at source line {src_line}: {body!r}"
                )
                out.append(src_lines[src_line - 1])
                src_line += 1
            elif tag == "-":
                assert src_lines[src_line - 1] == body, (
                    f"delete mismatch at source line {src_line}: {body!r}"
                )
                src_line += 1
            elif tag == "+":
                if hunk["old_start"] in selected:
                    out.append(body)
        # for skipped hunks: '+' lines are dropped, ' '/'-' consumed above
    while src_line <= len(src_lines):
        out.append(src_lines[src_line - 1])
        src_line += 1
    return "".join(out)


def main() -> int:
    for name in (V163, V182):
        if not name.exists():
            print(f"missing source: {name}")
            return 2
    src = V163.read_text(encoding="utf-8")
    dst = V182.read_text(encoding="utf-8")
    diff_text = DIFF.read_text(encoding="utf-8")
    hunks = parse_hunks(diff_text)
    starts = [h["old_start"] for h in hunks]
    print("hunk old_starts:", starts)
    assert set(starts) == APPLY_OLD_STARTS | SKIP_OLD_STARTS, "unexpected hunk set"

    full = apply_hunks(src, hunks, set(starts))
    if full != dst:
        print("SELF-CHECK FAILED: full-diff application does not reproduce v182")
        return 3
    print("self-check ok: v163 + full diff == v182 byte-exact")

    assembled = apply_hunks(src, hunks, APPLY_OLD_STARTS)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(assembled, encoding="utf-8", newline="")
    py_compile.compile(str(OUT), doraise=True)

    text = assembled
    missing = [n for n in API_NAMES if f"def {n}(" not in text]
    assert not missing, f"missing APIs: {missing}"
    assert text.count('def hif4_calibration_and_quantize_weight(') == 1
    assert text.count('def hif4_dynamic_quantize_q(') == 1
    assert "_WEIGHT_RESIDUAL_RANK = 2" in text
    assert "_ATTN_LOGIT_GAIN" not in text, "attention-side constants must not leak"
    assert "Official side-weight calibration (v163)" in text, "std attention block lost"
    assert "residual_u" in text and "rank1_u" in text

    print("written:", OUT)
    print("v163 sha256:", sha256(V163))
    print("v182 sha256:", sha256(V182))
    print("L2 sha256 :", sha256(OUT))
    return 0


if __name__ == "__main__":
    sys.exit(main())
