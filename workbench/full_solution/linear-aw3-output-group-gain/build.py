"""Build v206 from the retained v202 complete root."""

from pathlib import Path
import hashlib
import json


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
PARENT = ROOT / "solution.py"
CANDIDATE = HERE / "candidate" / "solution.py"
EXPECTED_PARENT_SHA256 = (
    "56dc805d6e5a3aef896db8021045740292735725d688b48e3d4393e55efcb2bd"
)

actual_parent_sha256 = hashlib.sha256(PARENT.read_bytes()).hexdigest()
if actual_parent_sha256 != EXPECTED_PARENT_SHA256:
    raise RuntimeError(
        f"expected retained v202 parent {EXPECTED_PARENT_SHA256}, "
        f"got {actual_parent_sha256}"
    )

source = PARENT.read_text(encoding="utf-8")
implementation = (HERE / "implementation.py").read_text(encoding="utf-8")
CANDIDATE.parent.mkdir(parents=True, exist_ok=True)
CANDIDATE.write_text(source + "\n\n" + implementation, encoding="utf-8")

candidate_sha256 = hashlib.sha256(CANDIDATE.read_bytes()).hexdigest()
config = {
    "run_id": "linear-aw3-output-group-gain",
    "version": "v206",
    "mechanism": "deployment-coordinate output-group x 64-block scalar A@W fit with hard re-encode output gate",
    "parent": "solution.py",
    "parent_sha256": actual_parent_sha256,
    "fit_data": "all calibration pairs and all rows supplied by eval-v3",
    "coordinate": "final natural deployment output groups and input blocks; gptq_block_order is activation-only",
    "output_group_size": 64,
    "candidate_count": 1,
    "official_status": "unregistered/NA",
}
(HERE / "config.json").write_text(
    json.dumps(config, indent=2) + "\n", encoding="utf-8"
)
print(json.dumps(config, indent=2))
