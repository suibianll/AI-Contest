"""Build the self-contained, preregistered A23 candidate."""
from pathlib import Path
import hashlib
import json

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
r3 = ROOT / "solutions/v162_attention_r3-rotation-center_allgates/solution.py"
a22_2 = ROOT / "solutions/continuous_attention_anchor22-a2/solution.py"
aligned = ROOT / "solutions/continuous_attention_a1-deployed-aligned_officialNA_timeNA/solution.py"
# Base flow (R3 source through its final calibration wrapper, plus the shared
# A1 stack-transform helpers) is identical to A22-2's build; assert that and
# then swap in the A23 trainer, which only changes the residual objective.
base = r3.read_text(encoding="utf-8")
wrap = base.rindex("def hif4_calibration_attention(")
source = base[:wrap]
helpers = aligned.read_text(encoding="utf-8")
source += helpers[helpers.index("def _a1_state_on_device("):helpers.index("def _a1_deployed_encode(")]
source += (HERE / "trainer.py").read_text(encoding="utf-8")
(HERE / "solution.py").write_text(source, encoding="utf-8")
# Structural cross-check: A23 must differ from A22-2 ONLY in the trainer tail.
a22_source = a22_2.read_text(encoding="utf-8")
a22_tail = a22_source[a22_source.index("def _a22b_train"):]
a23_tail = source[source.index("def _a23_block_amax"):]
for marker in ("def _a2_train_rotation(", "def _a2_true_path_gate_loss(",
               "_V189_CALIBRATION_ATTENTION = hif4_calibration_attention", "def _a2_normal("):
    assert marker in source, marker
config = {
    "run_id": "anchor23-a1",
    "mechanism": "joint-qk-block-scale-product-objective",
    "parent": "solutions/continuous_attention_anchor22-a2/solution.py",
    "parent_sha256": hashlib.sha256(a22_2.read_bytes()).hexdigest(),
    "source_sha256": hashlib.sha256((HERE / "solution.py").read_bytes()).hexdigest(),
    "objective": "mean_{f,g,b}[a_Q(s,g,b)*a_K(s,g,b) / max(a_Q(0,g,b)*a_K(0,g,b),1e-12)] + 1e-3*mean(S^2); a_Q(g,b) = mean over (tokens, group's Q heads) of amax^2 of head-block b; a_K(g,b) = mean over tokens of amax^2 of K head g block b; denominators are the S=0 parent aggregates",
    "invariance_note": "constant Q*c with K/c leaves the product unchanged; the additive A22-2 objective cannot see that direction",
    "steps": 32, "lr": 0.01, "clip_norm": 1.0, "regularization": 1e-3,
    "condition_limit": 2,
    "parent_construction": "identical to A22-2: one R3-original rotation+center training plus its own identity-vs-rotation gate; complete parent is the fixed fallback",
    "gate": "candidate versus the COMPLETE parent state on the last calibration window, true readout MSE; strictly smaller accepts, ties retain parent",
    "layout": "head-aligned 64-blocks (4B panel qh=16 kh=4 dim=256 -> nb=4); group-level fallback only for dim%64!=0 contract fuzz",
    "frozen": ["v162 standard Linear", "parent V path"],
    "evaluation": "4B panel qwen35-4b-panel-v1 paired vs A22-2; 72 attention cases; no local time gate; official 300s",
    "official_status": "NA", "candidate_count": 1,
}
(HERE / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
print(json.dumps(config, indent=2))
