"""Build C-FR1 by appending one block to the v237 complete root.

The parent is read from the immutable archive, never from the live root file,
which a parallel session can dirty -- the live root is cross-checked and
reported but never used as a build input.

The append is text, so the claim "the root is unchanged" is checkable rather
than assertable: the candidate's first `len(parent)` bytes must equal the parent
exactly, and the appended block is the only thing after them.
"""

from pathlib import Path
import ast
import hashlib
import json
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
PARENT = ROOT / "solutions/20260910_v237_linear-tf2-first-pass-gradient-reuse_scoreNA_timeNA/solution.py"
LIVE_ROOT = ROOT / "solution.py"
CANDIDATE = HERE / "candidate" / "solution.py"

EXPECTED_PARENT_SHA256 = (
    "ecb1f9e510b5507e1a2dc8b95f8a84e9b51864a420b3828537a328813e2ce554"
)
EXPECTED_PARENT_BYTES = 516697

sys.path.insert(0, str(HERE))
from cfr_block import CFR_BLOCK  # noqa: E402


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


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

    live_bytes = LIVE_ROOT.read_bytes()
    live_digest = sha256_bytes(live_bytes)

    module_bytes = CFR_BLOCK.strip().encode("utf-8")
    ast.parse(module_bytes.decode("utf-8"))  # the block must be valid Python

    candidate_bytes = (
        parent_bytes.rstrip(b"\n") + b"\n\n\n" + module_bytes + b"\n"
    )
    if not candidate_bytes.startswith(parent_bytes.rstrip(b"\n")):
        raise SystemExit("candidate is not an append of the parent bytes")
    # The parent carries two definitions of this name (an earlier stack at line
    # 9250 that `_V189_CALIBRATION_ATTENTION` captures, and the live one at line
    # 11529 that wins at module load); the append adds exactly one more.
    parent_defs = parent_bytes.count(b"def hif4_calibration_attention(")
    if candidate_bytes.count(b"def hif4_calibration_attention(") != parent_defs + 1:
        raise SystemExit(
            f"candidate should hold the parent's {parent_defs} definitions plus "
            "exactly one shadow"
        )
    if candidate_bytes.count(b"_CFR_PARENT_CALIBRATION = hif4_calibration_attention") != 1:
        raise SystemExit("the shadow must capture the parent calibration exactly once")

    tree = ast.parse(candidate_bytes.decode("utf-8"))
    module_level = [
        node.name for node in tree.body if isinstance(node, ast.FunctionDef)
    ]
    if module_level[-1] != "hif4_calibration_attention":
        raise SystemExit(
            "the shadow must be the last module-level definition, or the root's "
            "own calibration would win at import time"
        )
    if not module_level.count("hif4_calibration_attention") == parent_defs + 1:
        raise SystemExit("candidate must define hif4_calibration_attention once more")

    CANDIDATE.parent.mkdir(parents=True, exist_ok=True)
    CANDIDATE.write_bytes(candidate_bytes)

    info = {
        "parent_path": str(PARENT.relative_to(ROOT)).replace("\\", "/"),
        "parent_sha256": parent_digest,
        "parent_bytes": len(parent_bytes),
        "live_root_path": str(LIVE_ROOT.relative_to(ROOT)).replace("\\", "/"),
        "live_root_sha256": live_digest,
        "live_root_matches_archive": live_digest == parent_digest,
        "block_sha256": sha256_bytes(module_bytes),
        "block_bytes": len(module_bytes),
        "candidate_sha256": sha256_bytes(candidate_bytes),
        "candidate_bytes": len(candidate_bytes),
        "candidate_appended_bytes": len(candidate_bytes) - len(parent_bytes),
        "edit": "append one C-FR1 block: closed-form full-matrix reciprocal transform",
    }
    (HERE / "build.json").write_text(
        json.dumps(info, indent=2) + "\n", encoding="utf-8"
    )
    return info


if __name__ == "__main__":
    for key, value in main().items():
        print(f"{key}={value}")
