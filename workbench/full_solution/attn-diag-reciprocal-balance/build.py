"""Build the single, self-contained diagonal reciprocal-balance candidate."""

from pathlib import Path
import hashlib
import json


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
PARENT = ROOT / "solution.py"
CANDIDATE_DIR = HERE / "candidate"
CANDIDATE_DIR.mkdir(parents=True, exist_ok=True)

source = PARENT.read_text(encoding="utf-8")
signature = "    learned_center: Optional[torch.Tensor] = None,\n) -> dict[str, torch.Tensor]:\n"
replacement = "    learned_center: Optional[torch.Tensor] = None,\n    attention_diag_scale: Optional[torch.Tensor] = None,\n) -> dict[str, torch.Tensor]:\n"
if source.count(signature) != 1:
    raise RuntimeError("The _nvfp4_to_hif4 signature changed; rebuild needs review")
source = source.replace(signature, replacement, 1)

anchor = "    refine_importance = importance\n"
insertion = (
    "    if attention_diag_scale is not None:\n"
    "        diag_scale = _safe_positive_vector(\n"
    "            attention_diag_scale, channels\n"
    "        ).to(device=dense.device)\n"
    "        dense.mul_(diag_scale.reshape(*([1] * (dense.ndim - 1)), channels))\n"
    "    refine_importance = importance\n"
)
if source.count(anchor) != 1:
    raise RuntimeError("The _nvfp4_to_hif4 post-transform anchor changed")
source = source.replace(anchor, insertion, 1)

source += "\n\n" + (HERE / "implementation.py").read_text(encoding="utf-8")
candidate = CANDIDATE_DIR / "solution.py"
candidate.write_text(source, encoding="utf-8")

config = {
    "run_id": "attn-diag-reciprocal-balance",
    "mechanism": "closed-form-output-propagated-diagonal-reciprocal-qk-balance",
    "parent": "solution.py",
    "parent_sha256": hashlib.sha256(PARENT.read_bytes()).hexdigest(),
    "source_sha256": hashlib.sha256(candidate.read_bytes()).hexdigest(),
    "fit_windows": 3,
    "gate_windows": [4, 5],
    "max_tokens": 256,
    "chunk_tokens": 8,
    "formula": "d=0.25*log((b+1e-12)/(a+1e-12)); group-center; clamp +/-log(2)/2",
    "deployment": "Q_parent*exp(d), K_parent*exp(-d), learned_center*exp(-d)",
    "candidate_count": 1,
    "official_status": "unregistered/NA",
}
(HERE / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
print(json.dumps(config, indent=2))
