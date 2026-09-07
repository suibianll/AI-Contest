"""Build the self-contained, preregistered A21-1 candidate."""
from pathlib import Path
import hashlib
import json

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
parent = ROOT / "solutions/v162_attention_r3-rotation-center_allgates/solution.py"
aligned = ROOT / "solutions/continuous_attention_a1-deployed-aligned_officialNA_timeNA/solution.py"
source = parent.read_text(encoding="utf-8")
# Drop the old output trainer and its final calibration wrapper, preserving the
# exact deployed applier/attention reference and fixed orthogonal initialization.
start = source.index("def _a2_train_rotation(")
source = source[:start] + "\n_V189_CALIBRATION_ATTENTION = hif4_calibration_attention\n"
helpers = aligned.read_text(encoding="utf-8")
source += helpers[helpers.index("def _a1_state_on_device("):helpers.index("def _a1_deployed_encode(")]
source += (HERE / "trainer.py").read_text(encoding="utf-8")
(HERE / "solution.py").write_text(source, encoding="utf-8")
config = {
    "run_id": "anchor21-a1", "mechanism": "joint-block64-scale-inverse-transform",
    "parent": str(parent.relative_to(ROOT)), "parent_sha256": hashlib.sha256(parent.read_bytes()).hexdigest(),
    "source_sha256": hashlib.sha256((HERE / "solution.py").read_bytes()).hexdigest(),
    "steps": 32, "lr": 0.01, "clip_norm": 1.0, "regularization": 0.001,
    "epsilon": 1e-12, "amax_ties": "equal subgradient", "condition_limit": 2,
    "initialization": "R3 fixed pre-training Hadamard or identity; S=0",
    "folds": "all earlier calibration windows equally; last window true-output gate",
    "gate": "R3 PRE-TRAINING stack versus one learned scale candidate; ties retain stack",
    "evaluation_parent": "full official R3, never substitute the calibration fallback",
    "frozen": ["v162 standard Linear", "R3 V"],
    "official_status": "NA", "candidate_count": 1,
}
(HERE / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
print(json.dumps(config, indent=2))
