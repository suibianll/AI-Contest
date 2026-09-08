"""Build A4/v203 from the current v195 root."""

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
    "    a4_hierarchy_offsets: Optional[torch.Tensor] = None,\n"
    ") -> dict[str, torch.Tensor]:\n"
)
if source.count(signature) != 1:
    raise RuntimeError("_nvfp4_to_hif4 signature changed")
source = source.replace(signature, replacement, 1)

old_end = (
    "        params = _refine_activation_blocks64(dense, params, gram64)\n"
    "    return params\n"
    "\n"
    "\n"
    "def _dequantize_hif4(params: dict[str, torch.Tensor]) -> torch.Tensor:\n"
)
new_end = (
    "        params = _refine_activation_blocks64(dense, params, gram64)\n"
    "    if a4_hierarchy_offsets is not None:\n"
    "        params = _a4_apply_hierarchy_offsets(\n"
    "            dense, params, a4_hierarchy_offsets, refine_importance\n"
    "        )\n"
    "    return params\n"
    "\n"
    "\n"
    "def _dequantize_hif4(params: dict[str, torch.Tensor]) -> torch.Tensor:\n"
)
if source.count(old_end) != 1:
    raise RuntimeError("_nvfp4_to_hif4 return anchor changed")
source = source.replace(old_end, new_end, 1)

source += "\n\n" + (HERE / "implementation.py").read_text(encoding="utf-8")
candidate = CANDIDATE_DIR / "solution.py"
candidate.write_text(source, encoding="utf-8")

candidate_sha = hashlib.sha256(candidate.read_bytes()).hexdigest()
config = {
    "run_id": "attn-legal-hierarchy-selection",
    "version": "v203",
    "mechanism": "joint legal Q/K E6M2 scale-factor neighbor selection with one fixed adjacent code offset per candidate 64-block and exact legal lv2/lv3/mantissa re-encoding",
    "parent": "solution.py",
    "parent_sha256": actual_parent,
    "source_sha256": candidate_sha,
    "candidate_roles": "q-only, k-only, or joint qk per GQA group/block; directions +/-1 E6M2 code",
    "selection": "real causal Attention output MSE; at most one retained proposal per KV group",
    "deployment": "apply compiled per-64-block scale-code offset to each dynamic parent scale code, then solve legal hierarchy once",
    "continuous_transform": "frozen; no reciprocal coordinate, STE, optimizer, or dynamic search",
    "v_policy": "frozen",
    "candidate_count": 1,
    "official_status": "unregistered/NA",
}
(HERE / "config.json").write_text(
    json.dumps(config, indent=2) + "\n", encoding="utf-8"
)
print(json.dumps(config, indent=2))
