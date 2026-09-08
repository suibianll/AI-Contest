"""Build v198 attn-gqa-reciprocal-diag from the current root solution.py.

Patches _nvfp4_to_hif4 with an optional ``attention_diag_scale`` applied to
the final continuous tensor (after learned rotation and learned center,
right before importance/encode) and appends implementation.py, which adds
the Stage A analytic + smooth-max reciprocal diagonal compile, the Stage B
true-path hard gate on the last calibration window, and the q/k dynamic API
overrides that forward the per-group diag scale.  V is untouched.
"""

from pathlib import Path
import hashlib
import json


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
PARENT = ROOT / "solution.py"
CANDIDATE_DIR = HERE / "candidate"
CANDIDATE_DIR.mkdir(parents=True, exist_ok=True)

source = PARENT.read_text(encoding="utf-8")

PARENT_SHA256 = (
    "839adb1e617c3115c6b55071a34b281c5db0ff2aa070adbbc71fd1549e761d7f"
)
actual_parent_sha = hashlib.sha256(PARENT.read_bytes()).hexdigest()
if actual_parent_sha != PARENT_SHA256:
    raise RuntimeError(
        f"parent SHA256 {actual_parent_sha} != expected v195 root "
        f"{PARENT_SHA256}; rebuild needs review"
    )

signature = (
    "    learned_center: Optional[torch.Tensor] = None,\n"
    ") -> dict[str, torch.Tensor]:\n"
)
replacement = (
    "    learned_center: Optional[torch.Tensor] = None,\n"
    "    attention_diag_scale: Optional[torch.Tensor] = None,\n"
    ") -> dict[str, torch.Tensor]:\n"
)
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
    "run_id": "attn-gqa-reciprocal-diag",
    "version": "v198",
    "mechanism": "gqa-group-shared reciprocal diagonal exp(u): analytic RMS init + smooth-max range refine (tau=8, lambda=1e-3, 5 steps), true-path hard gate on last calibration window, V frozen",
    "parent": "solution.py",
    "parent_sha256": hashlib.sha256(PARENT.read_bytes()).hexdigest(),
    "source_sha256": hashlib.sha256(candidate.read_bytes()).hexdigest(),
    "u_bound": "log(2)",
    "tau": 8.0,
    "reg_lambda": 1.0e-3,
    "refine_steps": 5,
    "refine_lr": 0.25,
    "fit_windows": "all but last",
    "gate_window": "last",
    "deployment": "encode-stage attention_diag_scale: Q_final*exp(u), (K_final+c)*exp(-u); state field diag_scale",
    "candidate_count": 1,
    "official_status": "unregistered/NA",
}
(HERE / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
print(json.dumps(config, indent=2))
