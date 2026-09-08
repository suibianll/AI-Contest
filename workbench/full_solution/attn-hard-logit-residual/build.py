"""Build v201 from the current v195 root and the fixed v199 boundary helper."""

from pathlib import Path
import hashlib
import json


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
PARENT = ROOT / "solution.py"
V199 = ROOT / "workbench" / "full_solution" / "attn-gqa-hard-reciprocal"
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

v199_source = V199 / "implementation.py"
if not v199_source.is_file():
    raise RuntimeError(f"missing fixed v199 helper: {v199_source}")
source += "\n\n" + v199_source.read_text(encoding="utf-8")
source += "\n\n" + (HERE / "implementation.py").read_text(encoding="utf-8")

candidate = CANDIDATE_DIR / "solution.py"
candidate.write_text(source, encoding="utf-8")

candidate_sha = hashlib.sha256(candidate.read_bytes()).hexdigest()
config = {
    "run_id": "attn-hard-logit-residual",
    "version": "v201",
    "mechanism": "hard-logit residual weighted reciprocal proposals: v199 first +/- HiF4 boundary, top-2 residual ranking per GQA group, real causal hard-output selection, one block per group, V frozen",
    "parent": "solution.py",
    "parent_sha256": actual_parent,
    "helper_parent": "v199 fixed boundary generator",
    "source_sha256": candidate_sha,
    "u_bound": "+/-log(2)",
    "boundary_search": "v199 fixed 14-step binary locate of first changed hard code with 2e-4 crossing margin",
    "residual_weight": "local continuous-parent softmax Jacobian times frozen V deviation squared",
    "ranked_candidates": 2,
    "fit_windows": "all but last (fallback to all when only one exists)",
    "holdout_window": "last calibration window",
    "selection": "real causal Attention output MSE after residual ranking; at most one block move per KV/GQA group",
    "deployment": "one compiled diag_scale multiply after parent final Q/K transforms; K uses reciprocal scale",
    "v_policy": "frozen",
    "candidate_count": 1,
    "official_status": "unregistered/NA",
}
(HERE / "config.json").write_text(
    json.dumps(config, indent=2) + "\n", encoding="utf-8"
)
print(json.dumps(config, indent=2))
