"""Build the A-SR1 same-parent control: A-GR1 assembled on the v237 root.

The plan (section 3) requires this, and requires it explicitly rather than by
accident:

    "同父实现对照必须是这份新装的 A-GR1-on-v237，不是 v236——v236 建在旧父 v231
     上，直接拿它比较会混入不同Linear实现，使完整候选归因不单一"

v236 is the v231 root plus the A-GR1 block.  The v237 root is the v231 root plus
the L-TF2 shadow, so v236 is *not* a same-parent control for anything built on
v237: comparing against it would fold L-TF2's change into the attribution.

The assembly here is mechanical and derived:

    control = v237 bytes  +  the A-GR1 block taken byte-for-byte out of v236

The block is `v236[505762:]` -- everything after the v231 root prefix -- so it is
the same bytes that were reviewed and shipped as v236's A-GR1 section, just
appended to a different parent.  Nothing is retyped.

This control is **local only and is not submitted**.  It exists so the A-SR1
candidate has a same-parent baseline to be compared against bit for bit.

The script verifies, rather than assumes:

  * v237's bytes match the recorded SHA and byte count;
  * v236's first `505762` bytes are the v231 root, so the block boundary is the
    one claimed;
  * the assembled control starts with the v237 bytes and ends with the block;
  * the control holds exactly two `_agr1_train` definitions (v237 carries one
    already? no -- it holds exactly one, and the block adds one)... the script
    prints the counts it finds rather than asserting a number it has not checked.
"""

from pathlib import Path
import hashlib
import json


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]

V237 = ROOT / "solutions/20260910_v237_linear-tf2-first-pass-gradient-reuse_scoreNA_timeNA/solution.py"
V236 = ROOT / "solutions/20260910_v236_attention-agr1-on-v231_scoreNA_timeNA/solution.py"
V231 = ROOT / "solutions/20260910_v231_linear-em3-k2-arm_scoreNA_timeNA/solution.py"

OUT = HERE / "control" / "agr1-on-v237.py"
META = HERE / "control" / "agr1-on-v237.json"

V237_SHA256 = "ecb1f9e510b5507e1a2dc8b95f8a84e9b51864a420b3828537a328813e2ce554"
V236_SHA256 = "3319fc354af1ff0f35075498f26886bdf43a078d32a27852f575a4c50d32ba07"
V231_SHA256 = "ea79a1c12dc667142c620975aab188920fae7b29988c804f41c1a696cc5754f1"

V231_BYTES = 505762


def sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def main() -> dict:
    v237 = V237.read_bytes()
    v236 = V236.read_bytes()
    v231 = V231.read_bytes()

    if sha256(v237) != V237_SHA256:
        raise SystemExit("v237 archive is not the recorded bytes")
    if sha256(v236) != V236_SHA256:
        raise SystemExit("v236 archive is not the recorded bytes")
    if sha256(v231) != V231_SHA256:
        raise SystemExit("v231 archive is not the recorded bytes")
    if len(v231) != V231_BYTES:
        raise SystemExit("v231 archive is not the recorded length")
    if v236[:V231_BYTES] != v231:
        raise SystemExit("v236 does not start with the v231 root, so the block boundary is wrong")
    if not v237.startswith(v231):
        raise SystemExit("v237 does not start with the v231 root")

    block = v236[V231_BYTES:]
    if b"A-GR1" not in block:
        raise SystemExit("the block taken from v236 does not look like the A-GR1 section")

    control = v237.rstrip(b"\n") + b"\n\n\n" + block.lstrip(b"\n")

    if not control.startswith(v237.rstrip(b"\n")):
        raise SystemExit("control does not start with the v237 bytes")
    if control[V231_BYTES : V231_BYTES + (len(v237) - V231_BYTES)] != v237[V231_BYTES:]:
        raise SystemExit("control does not carry v237's L-TF2 section in place")
    if not control.endswith(block.lstrip(b"\n")):
        raise SystemExit("control does not end with the A-GR1 block")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_bytes(control)

    text = control.decode("utf-8")
    info = {
        "role": "local same-parent control for A-SR1; not submitted",
        "construction": "v237 bytes + the A-GR1 block taken byte-for-byte from v236",
        "v237_path": str(V237.relative_to(ROOT)).replace("\\", "/"),
        "v237_sha256": V237_SHA256,
        "v237_bytes": len(v237),
        "v231_root_prefix_bytes": V231_BYTES,
        "agr1_block_source": str(V236.relative_to(ROOT)).replace("\\", "/"),
        "agr1_block_sha256": sha256(block),
        "agr1_block_bytes": len(block),
        "control_path": str(OUT.relative_to(ROOT)).replace("\\", "/"),
        "control_sha256": sha256(control),
        "control_bytes": len(control),
        "carries_ltf2_shadow": b"if _pass:" in control,
        "definitions": {
            "hif4_calibration_attention": text.count("def hif4_calibration_attention("),
            "_agr1_train": text.count("def _agr1_train("),
            "_agr1_project": text.count("def _agr1_project("),
            "_em1_dynamic_descent": text.count("def _em1_dynamic_descent("),
        },
        "note": (
            "v236 is A-GR1 on the OLD parent v231 and is not a same-parent control "
            "for anything built on v237; it is kept as the cost-risk evidence for the "
            "original A-GR1 implementation, not as this card's baseline."
        ),
    }
    META.write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")
    return info


if __name__ == "__main__":
    for key, value in main().items():
        print(f"{key}={value}")
