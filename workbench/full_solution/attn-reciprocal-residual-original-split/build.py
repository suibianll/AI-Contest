"""Build the A22-2 original-split reciprocal-residual candidate from the current root."""

from pathlib import Path
import hashlib
import json


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
PARENT = ROOT / "solution.py"
CANDIDATE_DIR = HERE / "candidate"
CANDIDATE_DIR.mkdir(parents=True, exist_ok=True)

EXPECTED_PARENT_SHA256 = (
    "12352efdd4e23cc5e1e17953008664fbaa5ea5d693373635fdafc4d28ce4e24e"
)

source = PARENT.read_text(encoding="utf-8")
parent_sha = hashlib.sha256(PARENT.read_bytes()).hexdigest()
if parent_sha != EXPECTED_PARENT_SHA256:
    raise RuntimeError(f"Parent root SHA changed: {parent_sha}")
if source.count("def hif4_calibration_attention(") != 2:
    raise RuntimeError("Expected exactly two parent hif4_calibration_attention defs")
if source.count("_V189_CALIBRATION_ATTENTION = hif4_calibration_attention") != 1:
    raise RuntimeError("The v189 capture anchor changed; rebuild needs review")
for name in ("_a21_exp", "_a21_project", "_a22b_train", "_a21_gate_loss"):
    if f"def {name}(" in source:
        raise RuntimeError(f"Parent already defines {name}; name collision")

implementation = (HERE / "implementation.py").read_text(encoding="utf-8")
if implementation.count("def hif4_calibration_attention(") != 1:
    raise RuntimeError("Implementation must override hif4_calibration_attention once")
if "_A21_PARENT_CALIBRATION = hif4_calibration_attention" not in implementation:
    raise RuntimeError("Implementation lost its parent-capture anchor")

candidate = CANDIDATE_DIR / "solution.py"
candidate.write_text(source + "\n\n" + implementation, encoding="utf-8")

config = {
    "run_id": "attn-reciprocal-residual-original-split",
    "mechanism": "full-symmetric-zero-trace-reciprocal-qk-residual",
    "split": "A22-2 original: fit=calib_qkv_list[:-1], gate=calib_qkv_list[-1] only",
    "parent": "solution.py",
    "parent_sha256": parent_sha,
    "source_sha256": hashlib.sha256(candidate.read_bytes()).hexdigest(),
    "train_steps": 32,
    "learning_rate": 0.01,
    "gradient_clip": 1.0,
    "regularization": 0.001,
    "adam_beta": [0.9, 0.999],
    "spectral_bound": "+/-log(2)/2",
    "fit_windows": "all_but_last (4 of 5 on the official calibration panel)",
    "gate_windows": "last window only",
    "gate_rule": "single strict true-path output-MSE improvement, else parent fallback",
    "formula": "Q_new=Q_parent@exp(S), K_new=K_parent@exp(-S), trace(S)=0",
    "center_compile": "c_new=c_parent@exp(-S)",
    "candidate_count": 1,
    "official_status": "unregistered/NA",
}
(HERE / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
print(json.dumps(config, indent=2))
