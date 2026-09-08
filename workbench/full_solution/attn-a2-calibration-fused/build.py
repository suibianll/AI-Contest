"""Build v194 attn-a2-calibration-fused: A2/R3 calibration-equivalent speedup.

Three duplicate-computation eliminations, all bit-identical to the root:
1. Hoist _m_cayley_pair(theta) and rotation = base @ cayley out of the
   per-window training loop (theta is constant within one step).
2. Reuse std_v for v_hat in window preprocessing (same _dense_to_hif4(v_sub)).
3. Fuse the two _a2_true_path_gate_loss calls into one that returns
   (parent_mse, candidate_mse) with a single parent encode + forward.

solution.py has mixed line endings (CRLF + 193 lone LF), so all patching is
done on the raw text with explicit \r\n anchors and written back as bytes.
"""

from pathlib import Path
import hashlib
import json


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
PARENT = ROOT / "solution.py"
CANDIDATE_DIR = HERE / "candidate"
CANDIDATE_DIR.mkdir(parents=True, exist_ok=True)

source = PARENT.read_bytes().decode("utf-8")
NL = "\r\n"


def replace_once(old: str, new: str, label: str) -> None:
    global source
    if source.count(old) != 1:
        raise RuntimeError(f"anchor for {label} is not unique; rebuild needs review")
    source = source.replace(old, new, 1)


# --- Patch 1: hoist cayley/rotation out of the per-window training loop ------
old_train = NL.join([
    "    for step_index in range(_A2_TRAIN_STEPS):",
    "        window_losses = []",
    "        grad_theta = torch.zeros_like(theta)",
    "        for item in prepared:",
    "            c, _right = _m_cayley_pair(theta)",
    '            rotation = torch.einsum("dk,gkl->gdl", base, c)',
]) + NL
new_train = NL.join([
    "    for step_index in range(_A2_TRAIN_STEPS):",
    "        window_losses = []",
    "        grad_theta = torch.zeros_like(theta)",
    "        c, _right = _m_cayley_pair(theta)",
    '        rotation = torch.einsum("dk,gkl->gdl", base, c)',
    "        for item in prepared:",
]) + NL
replace_once(old_train, new_train, "patch1-hoist-cayley")

# --- Patch 2: v_hat reuses the std_v encode ----------------------------------
old_vhat = (
    "        v_hat = _dequantize_hif4(_dense_to_hif4(v_sub)).to(torch.float32)" + NL
)
new_vhat = "        v_hat = std_v" + NL
replace_once(old_vhat, new_vhat, "patch2-reuse-std_v")

# --- Patch 3a: fused gate loss returning (parent_mse, candidate_mse) ---------
old_gate = NL.join([
    "def _a2_true_path_gate_loss(",
    "    gate_window: dict,",
    "    q_heads: int,",
    "    kv_heads: int,",
    "    head_dim: int,",
    "    states: dict,",
    "    rotation: Optional[torch.Tensor],",
    "    device: torch.device,",
    "    center: Optional[torch.Tensor] = None,",
    ") -> float:",
    '    """Normalized full-window output loss through the true deployed path."""',
    "",
    "    def _run(state_q: dict, state_k: dict) -> float:",
    '        q_quant, q_scale = gate_window["q"]',
    '        k_quant, k_scale = gate_window["k"]',
    '        v_quant, v_scale = gate_window["v"]',
    "        q_ref = _dequantize_nvfp4_float32(q_quant, q_scale).to(torch.float32)",
    "        k_ref = _dequantize_nvfp4_float32(k_quant, k_scale).to(torch.float32)",
    "        v_ref = _dequantize_nvfp4_float32(v_quant, v_scale).to(torch.float32)",
    "        q_params = hif4_dynamic_quantize_q(",
    "            q_quant, q_scale, q_heads, head_dim, state_q",
    "        )",
    "        k_params = hif4_dynamic_quantize_k(",
    "            k_quant, k_scale, kv_heads, head_dim, state_k",
    "        )",
    "        v_params = hif4_dynamic_quantize_v(",
    "            v_quant, v_scale, kv_heads, head_dim, states[\"v_state\"]",
    "        )",
    "        q_hat = _dequantize_hif4(q_params).to(torch.float32)",
    "        k_hat = _dequantize_hif4(k_params).to(torch.float32)",
    "        v_hat = _dequantize_hif4(v_params).to(torch.float32)",
    "        reference = _a2_attention_forward(",
    "            q_ref[None], k_ref[None], v_ref[None], q_heads, kv_heads, head_dim",
    "        )[0]",
    "        player = _a2_attention_forward(",
    "            q_hat[None], k_hat[None], v_hat[None], q_heads, kv_heads, head_dim",
    "        )[0]",
    "        return float((player - reference).square().mean())",
    "",
    "    player_k_state = dict(states[\"k_state\"], learned_rotation=rotation)",
    "    if center is not None:",
    "        player_k_state[\"learned_center\"] = center",
    "    player_mse = _run(",
    "        dict(states[\"q_state\"], learned_rotation=rotation),",
    "        player_k_state,",
    "    )",
    "    standard_mse = _run(states[\"q_state\"], states[\"k_state\"])",
    "    return player_mse / max(standard_mse, 1e-12)",
]) + NL
new_gate = NL.join([
    "def _a2_true_path_gate_loss(",
    "    gate_window: dict,",
    "    q_heads: int,",
    "    kv_heads: int,",
    "    head_dim: int,",
    "    states: dict,",
    "    rotation: Optional[torch.Tensor],",
    "    device: torch.device,",
    "    center: Optional[torch.Tensor] = None,",
    ") -> Tuple[float, float]:",
    '    """Parent and candidate full-window output MSE through the true deployed path."""',
    "",
    '    q_quant, q_scale = gate_window["q"]',
    '    k_quant, k_scale = gate_window["k"]',
    '    v_quant, v_scale = gate_window["v"]',
    "    q_ref = _dequantize_nvfp4_float32(q_quant, q_scale).to(torch.float32)",
    "    k_ref = _dequantize_nvfp4_float32(k_quant, k_scale).to(torch.float32)",
    "    v_ref = _dequantize_nvfp4_float32(v_quant, v_scale).to(torch.float32)",
    "    reference = _a2_attention_forward(",
    "        q_ref[None], k_ref[None], v_ref[None], q_heads, kv_heads, head_dim",
    "    )[0]",
    "",
    "    def _player_mse(state_q: dict, state_k: dict) -> float:",
    "        q_params = hif4_dynamic_quantize_q(",
    "            q_quant, q_scale, q_heads, head_dim, state_q",
    "        )",
    "        k_params = hif4_dynamic_quantize_k(",
    "            k_quant, k_scale, kv_heads, head_dim, state_k",
    "        )",
    "        v_params = hif4_dynamic_quantize_v(",
    "            v_quant, v_scale, kv_heads, head_dim, states[\"v_state\"]",
    "        )",
    "        q_hat = _dequantize_hif4(q_params).to(torch.float32)",
    "        k_hat = _dequantize_hif4(k_params).to(torch.float32)",
    "        v_hat = _dequantize_hif4(v_params).to(torch.float32)",
    "        player = _a2_attention_forward(",
    "            q_hat[None], k_hat[None], v_hat[None], q_heads, kv_heads, head_dim",
    "        )[0]",
    "        return float((player - reference).square().mean())",
    "",
    "    parent_mse = _player_mse(states[\"q_state\"], states[\"k_state\"])",
    "    player_k_state = dict(states[\"k_state\"], learned_rotation=rotation)",
    "    if center is not None:",
    "        player_k_state[\"learned_center\"] = center",
    "    candidate_mse = _player_mse(",
    "        dict(states[\"q_state\"], learned_rotation=rotation),",
    "        player_k_state,",
    "    )",
    "    return parent_mse, candidate_mse",
]) + NL
replace_once(old_gate, new_gate, "patch3a-fused-gate-loss")

# --- Patch 3b: caller fuses identity/rotation gate into one call -------------
old_caller = NL.join([
    "        loss_identity = _a2_true_path_gate_loss(",
    "            gate_window, q_num_heads, kv_num_heads, head_dim, states, None, device",
    "        )",
    "        loss_rotation = _a2_true_path_gate_loss(",
    "            gate_window, q_num_heads, kv_num_heads, head_dim, states, rotation, device, center",
    "        )",
]) + NL
new_caller = NL.join([
    "        parent_mse, candidate_mse = _a2_true_path_gate_loss(",
    "            gate_window, q_num_heads, kv_num_heads, head_dim, states, rotation, device, center",
    "        )",
    "        loss_identity = parent_mse / max(parent_mse, 1e-12)",
    "        loss_rotation = candidate_mse / max(parent_mse, 1e-12)",
]) + NL
replace_once(old_caller, new_caller, "patch3b-fused-gate-caller")

candidate = CANDIDATE_DIR / "solution.py"
candidate.write_bytes(source.encode("utf-8"))

config = {
    "run_id": "attn-a2-calibration-fused",
    "version": "v194",
    "mechanism": "a2-calibration-equivalent-dedup: hoisted per-step cayley/rotation, shared std_v encode, fused parent/candidate gate",
    "parent": "solution.py",
    "parent_sha256": hashlib.sha256(PARENT.read_bytes()).hexdigest(),
    "source_sha256": hashlib.sha256(candidate.read_bytes()).hexdigest(),
    "algorithm_change": "none; bit-identical output contract",
    "candidate_count": 1,
    "official_status": "unregistered/NA",
}
(HERE / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
print(json.dumps(config, indent=2))
