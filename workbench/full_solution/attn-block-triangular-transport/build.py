"""Build the single, self-contained 64D block-triangular transport candidate."""

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
replacement = (
    "    learned_center: Optional[torch.Tensor] = None,\n"
    "    attention_triangular_blocks: Optional[torch.Tensor] = None,\n"
    "    attention_triangular_inverse: bool = False,\n"
    ") -> dict[str, torch.Tensor]:\n"
)
if source.count(signature) != 1:
    raise RuntimeError("The _nvfp4_to_hif4 signature changed; rebuild needs review")
source = source.replace(signature, replacement, 1)

anchor = "    refine_importance = importance\n"
insertion = (
    "    if attention_triangular_blocks is not None:\n"
    "        if learned_rotation_num_heads is None:\n"
    "            raise ValueError(\"Triangular transport requires head metadata\")\n"
    "        dense = _apply_attn_triangular_transform(\n"
    "            dense, int(learned_rotation_num_heads),\n"
    "            attention_triangular_blocks, bool(attention_triangular_inverse),\n"
    "        )\n"
    "    refine_importance = importance\n"
)
if source.count(anchor) != 1:
    raise RuntimeError("The _nvfp4_to_hif4 post-transform anchor changed")
source = source.replace(anchor, insertion, 1)

# Insert the helper before the new public wrappers are evaluated at runtime.
source += "\n\n" + (
    "def _apply_attn_triangular_transform(\n"
    "    dense: torch.Tensor,\n"
    "    num_heads: int,\n"
    "    blocks: torch.Tensor,\n"
    "    inverse: bool,\n"
    ") -> torch.Tensor:\n"
    "    matrices = blocks.detach().to(device=dense.device, dtype=torch.float32)\n"
    "    expected_rank = (matrices.ndim == 4 and int(matrices.shape[1]) == 2)\n"
    "    if not expected_rank or tuple(int(v) for v in matrices.shape[2:]) != (64, 64):\n"
    "        raise ValueError(\"Invalid triangular transport block shape\")\n"
    "    groups = int(matrices.shape[0])\n"
    "    if groups <= 0 or int(num_heads) % groups != 0:\n"
    "        raise ValueError(\"Triangular transport head counts are incompatible\")\n"
    "    if int(dense.shape[-1]) != int(num_heads) * 256:\n"
    "        raise ValueError(\"Triangular transport requires head_dim=256\")\n"
    "    per_head = matrices.repeat_interleave(int(num_heads) // groups, dim=0)\n"
    "    x = dense.to(torch.float32).reshape(*dense.shape[:-1], int(num_heads), 4, 64)\n"
    "    y = x.clone()\n"
    "    if not inverse:\n"
    "        y[..., 1, :] = x[..., 1, :] + torch.einsum(\"...hi,hij->...hj\", x[..., 0, :], per_head[:, 0])\n"
    "        y[..., 3, :] = x[..., 3, :] + torch.einsum(\"...hi,hij->...hj\", x[..., 2, :], per_head[:, 1])\n"
    "    else:\n"
    "        y[..., 0, :] = x[..., 0, :] - torch.einsum(\"...hj,hij->...hi\", x[..., 1, :], per_head[:, 0])\n"
    "        y[..., 2, :] = x[..., 2, :] - torch.einsum(\"...hj,hij->...hi\", x[..., 3, :], per_head[:, 1])\n"
    "    return y.reshape_as(dense)\n\n"
).replace("\n\ndef _apply_attn_triangular_transform", "\ndef _apply_attn_triangular_transform", 1)

# The helper text above is appended first; the implementation module contains
# the calibration routine and public Q/K/V wrappers and may call it later.
source += (HERE / "implementation.py").read_text(encoding="utf-8")

candidate = CANDIDATE_DIR / "solution.py"
candidate.write_text(source, encoding="utf-8")

config = {
    "run_id": "attn-block-triangular-transport",
    "mechanism": "two-rank1-64d-block-upper-triangular-qk-transport",
    "parent": "solution.py",
    "parent_sha256": hashlib.sha256(PARENT.read_bytes()).hexdigest(),
    "source_sha256": hashlib.sha256(candidate.read_bytes()).hexdigest(),
    "pairs": [[0, 1], [2, 3]],
    "block_dim": 64,
    "head_dim": 256,
    "fit_windows": [0, 1, 2],
    "gate_windows": [4, 5],
    "boundary_initial_step": 1.0e-5,
    "boundary_growth_steps": 16,
    "boundary_binary_steps": 10,
    "formula": "T=I+N; N_01=u0v0^T, N_23=u2v2^T; T_inverse=I-N",
    "deployment": "Q_parent*T, K_parent*T^{-T}; group-shared under GQA",
    "candidate_count": 1,
    "official_status": "unregistered/NA",
}
(HERE / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
print(json.dumps(config, indent=2))
