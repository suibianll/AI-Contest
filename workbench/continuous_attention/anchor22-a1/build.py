"""Build the self-contained, preregistered A22-1 candidate."""
from pathlib import Path
import hashlib
import json

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
parent = ROOT / "solutions/v162_attention_r3-rotation-center_allgates/solution.py"
aligned = ROOT / "solutions/continuous_attention_a1-deployed-aligned_officialNA_timeNA/solution.py"
source = parent.read_text(encoding="utf-8")
# Keep the ENTIRE R3 source up to (but excluding) its final calibration
# wrapper: the old rotation+center trainer, its true-path gate, the saved base
# calibration reference and _a2_normal all stay byte-identical. Only the final
# wrapper is replaced by the A22-1 trainer below.
wrap = source.rindex("def hif4_calibration_attention(")
source = source[:wrap]
helpers = aligned.read_text(encoding="utf-8")
source += helpers[helpers.index("def _a1_state_on_device("):helpers.index("def _a1_deployed_encode(")]
source += (HERE / "trainer.py").read_text(encoding="utf-8")
(HERE / "solution.py").write_text(source, encoding="utf-8")
config = {
    "run_id": "anchor22-a1",
    "mechanism": "fixed-scale-proposal-complete-r3-fallback",
    "parent": str(parent.relative_to(ROOT)),
    "parent_sha256": hashlib.sha256(parent.read_bytes()).hexdigest(),
    "source_sha256": hashlib.sha256((HERE / "solution.py").read_bytes()).hexdigest(),
    "proposal": "A21-1 joint block64 scale inverse transform, unchanged training function",
    "steps": 32, "lr": 0.01, "clip_norm": 1.0, "regularization": 0.001,
    "epsilon": 1e-12, "amax_ties": "equal subgradient", "condition_limit": 2,
    "initialization": "R3 fixed pre-training Hadamard or identity; S=0",
    "folds": "all earlier calibration windows equally; last window true-output gate",
    "parent_construction": "one R3-original rotation+center training plus its own identity-vs-rotation gate",
    "gate": "candidate versus the COMPLETE R3 parent state on the last calibration window, true readout MSE; strictly smaller accepts, ties retain parent",
    "fallback": "complete R3 states including learned_rotation and learned_center",
    "evaluation_parent": "full official R3, never substitute any calibration fallback",
    "frozen": ["v162 standard Linear", "R3 V"],
    "official_status": "NA", "candidate_count": 1,
}
(HERE / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
print(json.dumps(config, indent=2))
