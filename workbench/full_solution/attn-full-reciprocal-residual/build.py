"""Build the full-matrix reciprocal-residual candidate from the current root."""

from pathlib import Path
import hashlib
import json


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
PARENT = ROOT / "solution.py"
CANDIDATE_DIR = HERE / "candidate"
CANDIDATE_DIR.mkdir(parents=True, exist_ok=True)

candidate = CANDIDATE_DIR / "solution.py"
source = PARENT.read_text(encoding="utf-8")
source += "\n\n" + (HERE / "implementation.py").read_text(encoding="utf-8")
candidate.write_text(source, encoding="utf-8")

config = {
    "run_id": "attn-full-reciprocal-residual",
    "mechanism": "full-symmetric-zero-trace-reciprocal-qk-residual",
    "parent": "solution.py",
    "parent_sha256": hashlib.sha256(PARENT.read_bytes()).hexdigest(),
    "source_sha256": hashlib.sha256(candidate.read_bytes()).hexdigest(),
    "train_steps": 32,
    "learning_rate": 0.01,
    "gradient_clip": 1.0,
    "regularization": 0.001,
    "adam_beta": [0.9, 0.999],
    "spectral_bound": "+/-log(2)/2",
    "fit_windows": [0, 1, 2],
    "gate_windows": [4, 5],
    "formula": "Q_new=Q_parent@exp(S), K_new=K_parent@exp(-S), trace(S)=0",
    "center_compile": "c_new=c_parent@exp(-S)",
    "candidate_count": 1,
    "official_status": "unregistered/NA",
}
(HERE / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
print(json.dumps(config, indent=2))
