"""Build v199 from the current v195 root without importing repo modules."""

from pathlib import Path
import hashlib
import json


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
PARENT = ROOT / "solution.py"
CANDIDATE_DIR = HERE / "candidate"
CANDIDATE_DIR.mkdir(parents=True, exist_ok=True)

expected_parent = (
    "839adb1e617c3115c6b55071a34b281c5db0ff2aa070adbbc71fd1549e761d7f"
)
actual_parent = hashlib.sha256(PARENT.read_bytes()).hexdigest()
if actual_parent != expected_parent:
    raise RuntimeError(
        f"expected v195 parent {expected_parent}, got {actual_parent}"
    )

source = PARENT.read_text(encoding="utf-8")
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
    raise RuntimeError("_nvfp4_to_hif4 signature changed")
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
    raise RuntimeError("_nvfp4_to_hif4 post-transform anchor changed")
source = source.replace(anchor, insertion, 1)

source += "\n\n" + (HERE / "implementation.py").read_text(encoding="utf-8")
candidate = CANDIDATE_DIR / "solution.py"
candidate.write_text(source, encoding="utf-8")

candidate_sha = hashlib.sha256(candidate.read_bytes()).hexdigest()
config = {
    "run_id": "attn-gqa-hard-reciprocal",
    "version": "v199",
    "mechanism": "GQA-group-shared hard reciprocal coordinate: first +/- HiF4 boundary per 64-channel block, real causal Attention-output selection, one block per KV group, V frozen",
    "parent": "solution.py",
    "parent_sha256": actual_parent,
    "source_sha256": candidate_sha,
    "u_bound": "+/-log(2)",
    "boundary_search": "14-step binary locate of first changed hard code with fixed 2e-4 crossing margin",
    "fit_windows": "all but last (fallback to all when only one exists)",
    "holdout_window": "last calibration window",
    "selection": "real causal Attention output MSE; at most one block move per KV/GQA group",
    "deployment": "one compiled diag_scale multiply after parent final Q/K transforms; K uses reciprocal scale",
    "v_policy": "frozen",
    "candidate_count": 1,
    "official_status": "unregistered/NA",
}
(HERE / "config.json").write_text(
    json.dumps(config, indent=2) + "\n", encoding="utf-8"
)
print(json.dumps(config, indent=2))
