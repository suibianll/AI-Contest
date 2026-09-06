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
OUT = (
    Path(sys.argv[3])
    if len(sys.argv) > 3
    else ROOT / "workbench" / "v162_linear" / "candidate" / "solution.py"
)
DIFF = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("/tmp/v163_v182.diff")
TARGET = sys.argv[1] if len(sys.argv) > 1 else "l2"

# L2: v163 + the five Linear-side hunks of the v163->v182 diff (rank-2).
APPLY_OLD_STARTS = {71, 8155, 8218, 8429, 8473}
SKIP_OLD_STARTS = {365, 8761, 10188, 10295}
# L3: L2 + the pure-additive end-of-file v189 static-actorder block
# (v182->v189 diff hunk starting at old line 10783).  The one-line
# _DYNAMIC_OFFSETS change (old 498) is the v186 Attention-side edit and is
# NOT applied: the frozen Attention side must stay standard v162.
L3_APPEND_OLD_START = 10783

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
                if hunk["old_start"] in selected:
                    src_line += 1  # selected hunk: drop the deleted line
                else:
                    # skipped hunk: the deleted line stays part of the source
                    out.append(src_lines[src_line - 1])
                    src_line += 1
            elif tag == "+":
                if hunk["old_start"] in selected:
                    out.append(body)
        # for skipped hunks: '+' lines are dropped, ' '/'-' consumed above
    while src_line <= len(src_lines):
        out.append(src_lines[src_line - 1])
        src_line += 1
    return "".join(out)


def added_lines_of_hunk(hunks: list[dict], old_start: int) -> list[str]:
    hunk = next(h for h in hunks if h["old_start"] == old_start)
    return [line[1:] for line in hunk["lines"] if line.startswith("+")]


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

    if TARGET in ("l3", "l4"):
        diff189_path = ROOT / "workbench" / "v162_linear" / "v182_v189.diff"
        diff189 = parse_hunks(diff189_path.read_text(encoding="utf-8"))
        assert [h["old_start"] for h in diff189] == [498, L3_APPEND_OLD_START], (
            "v189 diff hunk set changed; reclassify before building"
        )
        # Only the end-of-file additive block is applied; the appended lines
        # are pure '+' (verified here), so appending to L2 is exact.
        appended = added_lines_of_hunk(diff189, L3_APPEND_OLD_START)
        raw189 = diff189_path.read_text(encoding="utf-8")
        body = raw189.split("@@ -10783,3 +10783,216 @@", 1)[1]
        assert not any(
            ln.startswith("-") for ln in body.splitlines()
            if ln[:1] in ("+", "-", " ") and not ln.startswith(("---", "+++"))
        ), "append hunk is not pure-additive; assemble by line numbers instead"
        assembled = assembled + "".join(appended)
        if not assembled.endswith("\n"):
            assembled += "\n"
        if TARGET == "l3":
            assert "_DYNAMIC_OFFSETS = (-1, 1, 2, 3)\n" in assembled, (
                "frozen-side discipline: v186 +4 window must not leak into L3"
            )
        else:
            # L4 = L3 + the v186 shared-constant widening (+4 E6M2 offset
            # code).  In this candidate the live Attention path is the
            # appended standard block, which never reads _DYNAMIC_OFFSETS,
            # so the widening only affects the Linear dynamic path — it is
            # required for an exact v189 Linear-side reproduction.
            v189_text = (ROOT / "solutions" / "20260906_v189_static-actorder-hdiag_recovered_scoreNA_timeNA" / "solution.py").read_text(encoding="utf-8")
            m = re.search(r"^_DYNAMIC_OFFSETS = .*$", v189_text, re.M)
            new_line = m.group(0)
            old_line = "_DYNAMIC_OFFSETS = (-1, 1, 2, 3)"
            assert assembled.count(old_line + "\n") == 1, "offset line not unique"
            assembled = assembled.replace(old_line + "\n", new_line + "\n", 1)
            assert new_line in assembled and "_DYNAMIC_OFFSETS = (-1, 1, 2, 3)\n" not in assembled

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(assembled, encoding="utf-8", newline="")
    py_compile.compile(str(OUT), doraise=True)

    text = assembled
    # Linear APIs: the base definition plus, for L3/L4, the appended override.
    n_linear_defs = text.count("def hif4_calibration_and_quantize_weight(")
    assert n_linear_defs == (2 if TARGET in ("l3", "l4") else 1), n_linear_defs
    # Attention APIs: two each (overridden v160 body + appended standard block).
    for attn_api in ("hif4_calibration_attention", "hif4_dynamic_quantize_q",
                     "hif4_dynamic_quantize_k", "hif4_dynamic_quantize_v"):
        assert text.count(f"def {attn_api}(") == 2, attn_api
    assert "_WEIGHT_RESIDUAL_RANK = 2" in text
    assert "_ATTN_LOGIT_GAIN" not in text, "attention-side constants must not leak"
    assert "Official side-weight calibration (v163)" in text, "std attention block lost"
    assert "residual_u" in text and "rank1_u" in text
    if TARGET in ("l3", "l4"):
        assert "_STATIC_ACTORDER_BASE_CALIBRATION" in text
        assert text.rstrip().endswith(")"), "unexpected tail"

    print("written:", OUT)
    print("v163 sha256:", sha256(V163))
    print("v182 sha256:", sha256(V182))
    print(f"{TARGET} sha256 :", sha256(OUT))
    return 0


if __name__ == "__main__":
    sys.exit(main())
