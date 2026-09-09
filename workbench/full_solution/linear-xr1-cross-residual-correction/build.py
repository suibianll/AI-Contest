"""Build L-XR1 from the retained v202 Linear + v195 Attention complete root.

Both L-XR1 hooks are API-level post-processors:

  * ``hif4_calibration_and_quantize_weight`` compiles the fixed rank-4 G/C
    factors after the parent weight/activation calibration returns;
  * ``hif4_dynamic_quantize_activation`` applies one frozen-gradient pass after
    the parent dynamic encoding returns.

Neither hook needs to reach inside a parent helper, so the candidate is the
retained root plus one appended module that rebinds those two names.  The
parent weight encoder, the shared HiF4 helpers, the Attention dynamic APIs and
every archived file stay byte-identical.

The build is byte-deterministic: the candidate is the parent bytes with the
implementation bytes appended after a fixed separator.
"""

from pathlib import Path
import hashlib


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
PARENT = ROOT / "solution.py"
IMPLEMENTATION = HERE / "implementation.py"
CANDIDATE = HERE / "candidate" / "solution.py"

EXPECTED_PARENT_SHA256 = (
    "56dc805d6e5a3aef896db8021045740292735725d688b48e3d4393e55efcb2bd"
)

REQUIRED_TOKENS = (
    "_XR1_PARENT_LINEAR_CALIBRATION = hif4_calibration_and_quantize_weight",
    "_XR1_PARENT_LINEAR_DYNAMIC = hif4_dynamic_quantize_activation",
    "def _xr1_compile_factors(",
    "def _xr1_dynamic_correct(",
    "def _xr1_top_eigen(",
    "def _xr1_top_singular(",
)


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def build() -> dict[str, str]:
    parent_bytes = PARENT.read_bytes()
    parent_digest = sha256_bytes(parent_bytes)
    if parent_digest != EXPECTED_PARENT_SHA256:
        raise SystemExit(
            f"parent SHA256 mismatch: {parent_digest} != {EXPECTED_PARENT_SHA256}"
        )

    implementation_bytes = IMPLEMENTATION.read_bytes()
    implementation_text = implementation_bytes.decode("utf-8")
    for token in REQUIRED_TOKENS:
        count = implementation_text.count(token)
        if count != 1:
            raise SystemExit(f"implementation token {token!r} occurs {count} times")

    candidate_bytes = (
        parent_bytes.rstrip(b"\n")
        + b"\n\n\n"
        + implementation_bytes.rstrip(b"\n")
        + b"\n"
    )
    CANDIDATE.parent.mkdir(parents=True, exist_ok=True)
    CANDIDATE.write_bytes(candidate_bytes)

    candidate_digest = sha256_bytes(candidate_bytes)
    return {
        "parent_sha256": parent_digest,
        "candidate_sha256": candidate_digest,
        "parent_bytes": str(len(parent_bytes)),
        "candidate_bytes": str(len(candidate_bytes)),
    }


if __name__ == "__main__":
    info = build()
    for key, value in info.items():
        print(f"{key}={value}")
