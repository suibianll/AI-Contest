"""Build the A-TG1 candidate by byte-level replacement of the v243 root.

Two contiguous replacements, each with a uniqueness assert and a head/tail
byte-identity check -- the `cmp`-prefix check alone does NOT prove a single
hunk (defect #37), so the check here is explicit.

    .venv/Scripts/python.exe build_v244.py
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "solution.py"
PARENT = ROOT / "solutions" / "20260911_v243_attention-atf1-cayley-hoist_scoreNA_timeNA" / "solution.py"

CRLF = "\r\n"


def lines(*rows: str) -> bytes:
    return CRLF.join(rows).encode() + CRLF.encode()


# --- replacement 1: lift the inner `_run` closure to a module-level arm function
OLD1 = lines(
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
    '            v_quant, v_scale, kv_heads, head_dim, states["v_state"]',
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
    '    player_k_state = dict(states["k_state"], learned_rotation=rotation)',
)

NEW1 = lines(
    "def _a2_gate_arm_mse(",
    "    gate_window: dict,",
    "    q_heads: int,",
    "    kv_heads: int,",
    "    head_dim: int,",
    "    states: dict,",
    "    state_q: dict,",
    "    state_k: dict,",
    ") -> float:",
    '    """One gate arm: full-window deployed-path MSE against the NVFP4 reference.',
    "",
    "    A-TG1 lifted this out of `_a2_true_path_gate_loss` so that the standard",
    "    (rotation-free) arm can be evaluated once and shared by both gate arms.",
    "    The body is the parent's `_run` closure verbatim, with the two states",
    "    passed in explicitly instead of closed over.",
    '    """',
    "",
    '    q_quant, q_scale = gate_window["q"]',
    '    k_quant, k_scale = gate_window["k"]',
    '    v_quant, v_scale = gate_window["v"]',
    "    q_ref = _dequantize_nvfp4_float32(q_quant, q_scale).to(torch.float32)",
    "    k_ref = _dequantize_nvfp4_float32(k_quant, k_scale).to(torch.float32)",
    "    v_ref = _dequantize_nvfp4_float32(v_quant, v_scale).to(torch.float32)",
    "    q_params = hif4_dynamic_quantize_q(",
    "        q_quant, q_scale, q_heads, head_dim, state_q",
    "    )",
    "    k_params = hif4_dynamic_quantize_k(",
    "        k_quant, k_scale, kv_heads, head_dim, state_k",
    "    )",
    "    v_params = hif4_dynamic_quantize_v(",
    '        v_quant, v_scale, kv_heads, head_dim, states["v_state"]',
    "    )",
    "    q_hat = _dequantize_hif4(q_params).to(torch.float32)",
    "    k_hat = _dequantize_hif4(k_params).to(torch.float32)",
    "    v_hat = _dequantize_hif4(v_params).to(torch.float32)",
    "    reference = _a2_attention_forward(",
    "        q_ref[None], k_ref[None], v_ref[None], q_heads, kv_heads, head_dim",
    "    )[0]",
    "    player = _a2_attention_forward(",
    "        q_hat[None], k_hat[None], v_hat[None], q_heads, kv_heads, head_dim",
    "    )[0]",
    "    return float((player - reference).square().mean())",
    "",
    "",
    "def _a2_true_path_gate_loss(",
    "    gate_window: dict,",
    "    q_heads: int,",
    "    kv_heads: int,",
    "    head_dim: int,",
    "    states: dict,",
    "    rotation: Optional[torch.Tensor],",
    "    device: torch.device,",
    "    center: Optional[torch.Tensor] = None,",
    "    standard_mse: Optional[float] = None,",
    ") -> float:",
    '    """Normalized full-window output loss through the true deployed path.',
    "",
    "    `standard_mse` lets a caller that has already evaluated the rotation-free",
    "    arm hand the value in; when omitted it is computed here, as before.",
    '    """',
    "",
    '    player_k_state = dict(states["k_state"], learned_rotation=rotation)',
)

# --- replacement 2: evaluate the standard arm once, share it with both gate arms
OLD2 = lines(
    "        loss_identity = _a2_true_path_gate_loss(",
    "            gate_window, q_num_heads, kv_num_heads, head_dim, states, None, device",
    "        )",
    "        loss_rotation = _a2_true_path_gate_loss(",
    "            gate_window, q_num_heads, kv_num_heads, head_dim, states, rotation, device, center",
    "        )",
)

NEW2 = lines(
    "        # A-TG1: the identity arm passes learned_rotation=None, so its player arm",
    "        # and its standard arm are the same computation on the same state --",
    "        # checked bit-equal on all six real attention layers -- which makes its",
    "        # value exactly x/max(x,1e-12) with x the standard arm.  Evaluating the",
    "        # standard arm once and sharing it drops the gate from four full-window",
    "        # arms to two.",
    "        standard_mse = _a2_gate_arm_mse(",
    "            gate_window, q_num_heads, kv_num_heads, head_dim, states,",
    '            states["q_state"], states["k_state"],',
    "        )",
    "        loss_identity = standard_mse / max(standard_mse, 1e-12)",
    "        loss_rotation = _a2_true_path_gate_loss(",
    "            gate_window, q_num_heads, kv_num_heads, head_dim, states, rotation, device, center,",
    "            standard_mse=standard_mse,",
    "        )",
)


# --- replacement 3: the tail of the gate function still calls the removed `_run`
OLD3 = lines(
    "    player_mse = _run(",
    '        dict(states["q_state"], learned_rotation=rotation),',
    "        player_k_state,",
    "    )",
    '    standard_mse = _run(states["q_state"], states["k_state"])',
    "    return player_mse / max(standard_mse, 1e-12)",
)

NEW3 = lines(
    "    player_mse = _a2_gate_arm_mse(",
    "        gate_window, q_heads, kv_heads, head_dim, states,",
    '        dict(states["q_state"], learned_rotation=rotation),',
    "        player_k_state,",
    "    )",
    "    if standard_mse is None:",
    "        standard_mse = _a2_gate_arm_mse(",
    "            gate_window, q_heads, kv_heads, head_dim, states,",
    '            states["q_state"], states["k_state"],',
    "        )",
    "    return player_mse / max(standard_mse, 1e-12)",
)


def replace_once(blob: bytes, old: bytes, new: bytes, tag: str) -> bytes:
    hits = blob.count(old)
    if hits != 1:
        raise SystemExit(f"{tag}: expected exactly 1 occurrence, found {hits} -- refusing")
    k = blob.index(old)
    out = blob[:k] + new + blob[k + len(old):]
    if out[:k] != blob[:k] or out[k + len(new):] != blob[k + len(old):]:
        raise SystemExit(f"{tag}: head/tail byte-identity FAILED")
    print(f"  {tag}: ok  (pos {k}, {len(old)} -> {len(new)} B)")
    return out


def main() -> int:
    src = PARENT.read_bytes()
    parent_sha = hashlib.sha256(src).hexdigest()
    print(f"parent: {len(src)} B  sha256 {parent_sha.upper()}")
    if parent_sha != "86169fd57b417e73cba307c1d5c8ecb326d480e25cc447f643da2c79cc8d82ea":
        print("  NOTE: parent is not the v243 archive SHA; proceeding but recording it.")
    out = replace_once(src, OLD1, NEW1, "replace-1 lift _run")
    out = replace_once(out, OLD3, NEW3, "replace-3 rewired tail")
    out = replace_once(out, OLD2, NEW2, "replace-2 share standard arm")
    SRC.write_bytes(out)
    print(f"candidate: {len(out)} B  sha256 {hashlib.sha256(out).hexdigest().upper()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
