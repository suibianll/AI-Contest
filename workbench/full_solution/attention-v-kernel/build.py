"""Build the VK candidate by appending one block to the v237 complete root.

Parent is read from the immutable archive, never from the live root file.
The append is text, so "the root is unchanged" is checked rather than asserted.
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
from vk_candidate_block import VK_BLOCK  # noqa: E402


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def main() -> dict:
    parent_bytes = PARENT.read_bytes()
    parent_digest = sha256_bytes(parent_bytes)
    if parent_digest != EXPECTED_PARENT_SHA256:
        raise SystemExit(f"parent SHA256 mismatch: {parent_digest}")
    if len(parent_bytes) != EXPECTED_PARENT_BYTES:
        raise SystemExit(f"parent byte count mismatch: {len(parent_bytes)}")

    module_bytes = VK_BLOCK.strip().encode("utf-8")
    ast.parse(module_bytes.decode("utf-8"))

    candidate_bytes = parent_bytes.rstrip(b"\n") + b"\n\n\n" + module_bytes + b"\n"
    if not candidate_bytes.startswith(parent_bytes.rstrip(b"\n")):
        raise SystemExit("candidate is not an append of the parent bytes")

    # The shadow must be the LAST module-level definition of each name, or the
    # root's own version would win at import time.
    tree = ast.parse(candidate_bytes.decode("utf-8"))
    names = [n.name for n in tree.body if isinstance(n, ast.FunctionDef)]
    if names[-1] != "hif4_dynamic_quantize_v":
        raise SystemExit(f"last module-level def is {names[-1]}, expected the V shadow")

    for name in ("hif4_calibration_attention", "hif4_dynamic_quantize_v"):
        if names.count(name) < 2:
            raise SystemExit(f"{name} shadow missing")

    CANDIDATE.parent.mkdir(parents=True, exist_ok=True)
    CANDIDATE.write_bytes(candidate_bytes)

    live_digest = sha256_bytes(LIVE_ROOT.read_bytes())
    info = {
        "parent_path": str(PARENT.relative_to(ROOT)).replace("\\", "/"),
        "parent_sha256": parent_digest,
        "parent_bytes": len(parent_bytes),
        "live_root_matches_archive": live_digest == parent_digest,
        "block_bytes": len(module_bytes),
        "candidate_sha256": sha256_bytes(candidate_bytes),
        "candidate_bytes": len(candidate_bytes),
        "appended_bytes": len(candidate_bytes) - len(parent_bytes),
        "edit": "append the VK block: gated relative-position kernel + V code correction",
    }
    (HERE / "build.json").write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")
    return info


if __name__ == "__main__":
    for key, value in main().items():
        print(f"{key}={value}")
