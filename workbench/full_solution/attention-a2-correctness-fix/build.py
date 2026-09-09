"""Build the A2 correctness-fix candidate from the retained v202 root."""

from pathlib import Path
import hashlib
import json


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
PARENT = ROOT / "solution.py"
CANDIDATE = HERE / "candidate" / "solution.py"
EXPECTED_PARENT_SHA256 = "56dc805d6e5a3aef896db8021045740292735725d688b48e3d4393e55efcb2bd"


def replace_once(source: str, old: str, new: str, label: str) -> str:
    count = source.count(old)
    if count != 1:
        raise RuntimeError(f"{label}: expected one source match, found {count}")
    return source.replace(old, new, 1)


actual_parent_sha256 = hashlib.sha256(PARENT.read_bytes()).hexdigest()
if actual_parent_sha256 != EXPECTED_PARENT_SHA256:
    raise RuntimeError(
        f"expected retained v202 parent {EXPECTED_PARENT_SHA256}, "
        f"got {actual_parent_sha256}"
    )

source = PARENT.read_text(encoding="utf-8")
source = replace_once(
    source,
    ") -> Tuple[torch.Tensor, dict]:\n"
    "    \"\"\"Analytic-gradient trainer: no autograd, safe under any harness context.\"\"\"",
    ") -> Tuple[torch.Tensor, dict, torch.Tensor]:\n"
    "    \"\"\"Analytic-gradient trainer: no autograd, safe under any harness context.\"\"\"",
    "A2 return annotation",
)
source = replace_once(
    source,
    "        data_loss = torch.stack(window_losses).mean()\n"
    "        if not math.isfinite(float(data_loss)):\n"
    "            raise RuntimeError(\"A2 rotation training produced a non-finite loss\")\n"
    "        c_now, _right = _m_cayley_pair(theta)",
    "        data_loss = torch.stack(window_losses).mean()\n"
    "        if not math.isfinite(float(data_loss)):\n"
    "            raise RuntimeError(\"A2 rotation training produced a non-finite loss\")\n"
    "        window_count = float(len(prepared))\n"
    "        grad_theta = grad_theta / window_count\n"
    "        grad_center = grad_center / window_count\n"
    "        c_now, _right = _m_cayley_pair(theta)",
    "multi-window gradient normalization",
)
function_start = source.index("def hif4_calibration_attention(\n", source.index("_V189_CALIBRATION_ATTENTION"))
function_end = source.index("\n\n\n# ---------------------------------------------------------------------------", function_start)
old_function = source[function_start:function_end]
new_function = '''def hif4_calibration_attention(
    calib_qkv_list: list,
    q_num_heads: int,
    kv_num_heads: int,
    head_dim: int,
) -> dict[str, Any]:
    """v189 stack calibration + gated learned rotation (A2 frozen config)."""

    states = _V189_CALIBRATION_ATTENTION(
        calib_qkv_list, q_num_heads, kv_num_heads, head_dim
    )
    if not isinstance(calib_qkv_list, list) or len(calib_qkv_list) < 2:
        return states

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    windows = [
        {
            "q": _dequantize_nvfp4_float32(*item["q"]).to(torch.float32),
            "k": _dequantize_nvfp4_float32(*item["k"]).to(torch.float32),
            "v": _dequantize_nvfp4_float32(*item["v"]).to(torch.float32),
        }
        for item in calib_qkv_list
    ]
    rotation, info, center = _a2_train_rotation(
        windows[:-1], q_num_heads, kv_num_heads, head_dim, device
    )
    gate_window = calib_qkv_list[-1]
    loss_identity = _a2_true_path_gate_loss(
        gate_window, q_num_heads, kv_num_heads, head_dim, states, None, device
    )
    loss_rotation = _a2_true_path_gate_loss(
        gate_window, q_num_heads, kv_num_heads, head_dim, states, rotation, device, center
    )
    if loss_rotation < loss_identity:
        cpu_rotation = rotation.detach().cpu().to(torch.float32).clone()
        states["q_state"]["learned_rotation"] = cpu_rotation
        states["k_state"]["learned_rotation"] = cpu_rotation.clone()
        states["k_state"]["learned_center"] = center
        audit = {
            "a2_arm": "rotation",
            "a2_gate_loss_identity": float(loss_identity),
            "a2_gate_loss_rotation": float(loss_rotation),
            "a2_steps": int(info["steps"]),
            "a2_train_loss": float(info["final_train_loss"]),
            "a2_ortho_error": float(info["ortho_error"]),
        }
    else:
        audit = {
            "a2_arm": "identity",
            "a2_gate_loss_identity": float(loss_identity),
            "a2_gate_loss_rotation": float(loss_rotation),
            "a2_steps": int(info["steps"]),
            "a2_train_loss": float(info["final_train_loss"]),
            "a2_ortho_error": float(info["ortho_error"]),
        }
    states["q_state"].update(audit)
    states["k_state"].update(audit)
    return states'''
source = source[:function_start] + new_function + source[function_end:]

CANDIDATE.parent.mkdir(parents=True, exist_ok=True)
CANDIDATE.write_text(source, encoding="utf-8")
candidate_sha256 = hashlib.sha256(CANDIDATE.read_bytes()).hexdigest()
config = {
    "run_id": "attention-a2-correctness-fix",
    "version": "unassigned-until-material-output-change",
    "mechanism": "A2 objective scaling and error visibility correction",
    "parent": "solution.py",
    "parent_sha256": actual_parent_sha256,
    "candidate_sha256": candidate_sha256,
    "changes": [
        "mean gradients over training windows",
        "correct three-value return annotation",
        "propagate calibration and A2 implementation errors",
    ],
    "official_status": "unregistered/NA",
}
(HERE / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
print(json.dumps(config, indent=2))
