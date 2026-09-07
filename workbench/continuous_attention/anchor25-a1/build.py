"""Build the self-contained, preregistered A25 candidate."""
from pathlib import Path
import hashlib
import json

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
r3 = ROOT / "solutions/v162_attention_r3-rotation-center_allgates/solution.py"
aligned = ROOT / "solutions/continuous_attention_a1-deployed-aligned_officialNA_timeNA/solution.py"
# Base flow: R3 source through its final calibration wrapper (contains the
# old trainer we DO NOT call, plus all shared helpers), plus A1 stack-transform
# helpers. The A25 trainer replaces the final wrapper with single-trunk
# scale-proxy training on B coordinates (no R3 old training, no Hadamard).
base = r3.read_text(encoding="utf-8")
wrap = base.rindex("def hif4_calibration_attention(")
source = base[:wrap]
helpers = aligned.read_text(encoding="utf-8")
source += helpers[helpers.index("def _a1_state_on_device("):helpers.index("def _a1_deployed_encode(")]
source += (HERE / "trainer.py").read_text(encoding="utf-8")
(HERE / "solution.py").write_text(source, encoding="utf-8")
for marker in ("_V189_CALIBRATION_ATTENTION = hif4_calibration_attention", "def _a2_normal(",
               "def _a1_stack_transform(", "def _a25_train(", "a25_gate_base_mse"):
    assert marker in source, marker
config = {
    "run_id": "anchor25-a1",
    "mechanism": "single-trunk-inverse-scale-proxy-on-base-stack",
    "parent": "base stack B (v189 _V189_CALIBRATION_ATTENTION, no learned_rotation/center)",
    "parent_sha256": "261202248a0146a2ee45f3df60bd1979bb8171b7c162921013b0024c848617af",
    "source_sha256": hashlib.sha256((HERE / "solution.py").read_bytes()).hexdigest(),
    "trunk": "single: no R3 old training, no Hadamard pre-rotation, S=0 identity start",
    "objective": "A22-2 additive scale proxy (officially validated +19): per-role 64-block amax ratio squared, Q/K block-mean summed, equal-weight folds, epsilon/amax-tie subgradient",
    "no_ste": "STE output-error objective proven zero-gradient under inverse parameterization (A24); not used",
    "steps": 32, "lr": 0.01, "clip_norm": 1.0, "regularization": 1e-3,
    "gate": "candidate versus BASE stack B on the last calibration window, true readout MSE; strictly smaller accepts, ties retain B",
    "diagnostics": "four-dimension: scale ratio / code change count / QK MSE / Attention output MSE; three-way: B / R3(P_old) / C",
    "dedup": "A21-1 (Hadamard pre-rotation, no diagnostics, no three-way); A22-2 (retains R3 old training + residual); A23 (retains + product objective); A24 (STE, zero gradient)",
    "frozen": ["v162 standard Linear", "parent V path"],
    "evaluation": "4B panel paired vs B and vs R3 (three-way); 72 attention cases; no local time gate; official 300s; local paired is risk record only (A23 precedent)",
    "official_status": "NA", "candidate_count": 1,
}
(HERE / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
print(json.dumps(config, indent=2))
