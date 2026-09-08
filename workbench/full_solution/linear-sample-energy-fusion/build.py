"""Build v202 from the current v195 root with fused sample-energy compilation."""

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
old_lists = (
    "    activation_samples: list[torch.Tensor] = []\n"
    "    activation_e2e_samples: list[torch.Tensor] = []\n"
)
new_lists = (
    "    activation_samples: list[torch.Tensor] = []\n"
    "    activation_e2e_samples: list[torch.Tensor] = []\n"
    "    activation_full_samples: list[torch.Tensor] = []\n"
)
if source.count(old_lists) != 1:
    raise RuntimeError("activation sample declarations changed")
source = source.replace(old_lists, new_lists, 1)

old_validation = (
    "        if activation.ndim != 2 or int(activation.shape[1]) != in_features:\n"
    "            raise ValueError(\"Calibration activation shape is incompatible with weight\")\n"
    "        stats_sample = _sample_rows(activation, _LINEAR_STATS_TOKENS)\n"
)
new_validation = (
    "        if activation.ndim != 2 or int(activation.shape[1]) != in_features:\n"
    "            raise ValueError(\"Calibration activation shape is incompatible with weight\")\n"
    "        activation_full_samples.append(activation)\n"
    "        stats_sample = _sample_rows(activation, _LINEAR_STATS_TOKENS)\n"
)
if source.count(old_validation) != 1:
    raise RuntimeError("activation decode loop changed")
source = source.replace(old_validation, new_validation, 1)

old_transform_anchor = (
    "    smooth_inv_work = None if smooth_inv_state is None else best_d.reciprocal()\n"
    "    permutation_work = None if permutation_state is None else best_perm\n"
    "\n"
    "    # v166: transformed_activation_samples are built before the weight\n"
)
new_transform_anchor = (
    "    smooth_inv_work = None if smooth_inv_state is None else best_d.reciprocal()\n"
    "    permutation_work = None if permutation_state is None else best_perm\n"
    "    compiled_sample_energy_order = _v202_sample_energy_block_order_from_calibration(\n"
    "        activation_full_samples,\n"
    "        smooth_inv_work,\n"
    "        permutation_work,\n"
    "        best_block_smooth_size,\n"
    "        best_block_smooth_seed,\n"
    "        residual_u,\n"
    "        residual_v,\n"
    "        rank1_u,\n"
    "        rank1_v,\n"
    "        activation_importance,\n"
    "    )\n"
    "\n"
    "    # v166: transformed_activation_samples are built before the weight\n"
)
if source.count(old_transform_anchor) != 1:
    raise RuntimeError("final transform anchor changed")
source = source.replace(old_transform_anchor, new_transform_anchor, 1)

old_state = "        \"smooth_inv\": smooth_inv_state,\n"
new_state = (
    "        \"smooth_inv\": smooth_inv_state,\n"
    "        \"compiled_sample_energy_order\": compiled_sample_energy_order,\n"
)
if source.count(old_state) != 1:
    raise RuntimeError("activation state anchor changed")
source = source.replace(old_state, new_state, 1)

source += "\n\n" + (HERE / "implementation.py").read_text(encoding="utf-8")
candidate = CANDIDATE_DIR / "solution.py"
candidate.write_text(source, encoding="utf-8")

candidate_sha = hashlib.sha256(candidate.read_bytes()).hexdigest()
config = {
    "run_id": "linear-sample-energy-fusion",
    "version": "v202",
    "mechanism": "fuse existing sample-energy block-order compilation with the parent calibration's first decoded activation tensors",
    "parent": "solution.py",
    "parent_sha256": actual_parent,
    "source_sha256": candidate_sha,
    "equivalence_target": "current combined Linear root: same final deployment-coordinate sample-energy order",
    "removed_work": "post-calibration _dequantize_nvfp4_float32 plus _static_actorder_dense_from_state rebuild",
    "dynamic_path": "unchanged; consumes the compiled gptq_block_order",
    "candidate_count": 1,
    "official_status": "unregistered/NA",
}
(HERE / "config.json").write_text(
    json.dumps(config, indent=2) + "\n", encoding="utf-8"
)
print(json.dumps(config, indent=2))
