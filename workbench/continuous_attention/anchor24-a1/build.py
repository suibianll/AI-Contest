"""Build the self-contained, preregistered A24 candidate."""
from pathlib import Path
import hashlib
import json

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
r3 = ROOT / "solutions/v162_attention_r3-rotation-center_allgates/solution.py"
a22_2 = ROOT / "solutions/continuous_attention_anchor22-a2/solution.py"
aligned = ROOT / "solutions/continuous_attention_a1-deployed-aligned_officialNA_timeNA/solution.py"
# Base flow (R3 source through its final calibration wrapper, plus the shared
# A1 stack-transform helpers) is identical to A22-2/A23's build; only the
# trainer tail is swapped, and the A24 tail replaces the residual objective
# with the true quantized attention output error.
base = r3.read_text(encoding="utf-8")
wrap = base.rindex("def hif4_calibration_attention(")
source = base[:wrap]
helpers = aligned.read_text(encoding="utf-8")
source += helpers[helpers.index("def _a1_state_on_device("):helpers.index("def _a1_deployed_encode(")]
source += (HERE / "trainer.py").read_text(encoding="utf-8")
(HERE / "solution.py").write_text(source, encoding="utf-8")
for marker in ("def _a2_train_rotation(", "def _a2_true_path_gate_loss(",
               "_V189_CALIBRATION_ATTENTION = hif4_calibration_attention", "def _a2_normal(",
               "_A2_MAX_KV_TOKENS = 128", "_A2_MAX_Q_TOKENS = 32", "def _m_attention_backward("):
    assert marker in source, marker
config = {
    "run_id": "anchor24-a1",
    "mechanism": "residual-inverse-transform-true-output-error-objective",
    "parent": "solutions/continuous_attention_anchor22-a2/solution.py",
    "parent_sha256": hashlib.sha256(a22_2.read_bytes()).hexdigest(),
    "source_sha256": hashlib.sha256((HERE / "solution.py").read_bytes()).hexdigest(),
    "objective": "true deployed-path attention output MSE vs the NVFP4 reference, A2-style mse_std normalization, window-equal mean; straight-through manual gradients (_m_attention_backward + pre-encode einsum); per-window gradient SUM then clip1 (R3 trainer convention)",
    "no_scale_regularization": "the error target is the objective; complexity is bounded by 32 steps and the spectral projection (zero-trace, +/-log2/2, cond<=2)",
    "steps": 32, "lr": 0.01, "clip_norm": 1.0,
    "parent_construction": "identical to A22-2: one R3-original rotation+center training plus its own identity-vs-rotation gate; complete parent is the fixed fallback",
    "gate": "candidate versus the COMPLETE parent state on the last calibration window, true readout MSE; strictly smaller accepts, ties retain parent",
    "dedup": "v161/v128 per-call dynamic refinement (dynamic-side, timeout family) - A24 is calibration-side only; v187 Jacobian importance (static analytic selection, no iterative training); A2 trainer (same objective form but Cayley theta + independent center at the B coordinates vs symmetric inverse exp(+-S) + compiled center at the P coordinates, orthogonal vs inverse-pair deployment class, parent R3 vs A22-2); A21/A22/A23 (scale-proxy objectives)",
    "frozen": ["v162 standard Linear", "parent V path"],
    "evaluation": "4B panel qwen35-4b-panel-v1 paired vs A22-2 (baseline already filled by anchor23-a1 run, mean 0.536715); 72 attention cases; no local time gate; official 300s",
    "official_status": "NA", "candidate_count": 1,
}
(HERE / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
print(json.dumps(config, indent=2))
