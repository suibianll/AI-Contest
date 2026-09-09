"""Build the A-H1 threshold-event-search candidate from the retained v202 root."""

from pathlib import Path
import hashlib
import json


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
PARENT = ROOT / "solution.py"
CANDIDATE = HERE / "candidate" / "solution.py"
AH1_CODE = HERE / "ah1_code.py"
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

# 1) Snapshot theta/center just before the final Adam update (no arithmetic
#    change to the root trainer; the deployed rotation/center are untouched).
source = replace_once(
    source,
    "    eye = torch.eye(dim, device=device, dtype=torch.float32)\n"
    "    final_loss = float(\"nan\")\n",
    "    eye = torch.eye(dim, device=device, dtype=torch.float32)\n"
    "    final_loss = float(\"nan\")\n"
    "    theta_pre = None\n"
    "    center_pre = None\n"
    "    ah1_g_theta = None\n"
    "    ah1_g_center = None\n",
    "A-H1 pre-update snapshot init",
)
source = replace_once(
    source,
    "        total = grad_theta.norm()\n"
    "        if float(total) > _A2_TRAIN_CLIP and float(total) > 0:\n"
    "            grad_theta = grad_theta * (_A2_TRAIN_CLIP / float(total))\n"
    "        exp_avg = 0.9 * exp_avg + 0.1 * grad_theta\n",
    "        total = grad_theta.norm()\n"
    "        if float(total) > _A2_TRAIN_CLIP and float(total) > 0:\n"
    "            grad_theta = grad_theta * (_A2_TRAIN_CLIP / float(total))\n"
    "        if step_index == _A2_TRAIN_STEPS - 1:\n"
    "            theta_pre = theta.clone()\n"
    "            center_pre = center.clone()\n"
    "        exp_avg = 0.9 * exp_avg + 0.1 * grad_theta\n",
    "A-H1 pre-update snapshot",
)

# 2) After training, recompute the window-mean gradient at the snapshot point
#    (data gradients divided by the window count, then the same reg term; no
#    clip and no Adam - this only defines the A-H1 search direction).
mean_gradient_pass = (
    "    if theta_pre is not None and prepared:\n"
    "        m_grad_theta = torch.zeros_like(theta)\n"
    "        m_grad_center = torch.zeros_like(center)\n"
    "        for item in prepared:\n"
    "            m_c, _m_right = _m_cayley_pair(theta_pre)\n"
    "            m_rotation = torch.einsum(\"dk,gkl->gdl\", base, m_c)\n"
    "            m_q_rot = _a2_apply_group_rotation(item[\"q\"], q_heads, m_rotation)\n"
    "            m_k_rot = _a2_apply_group_rotation(item[\"k\"], kv_heads, m_rotation)\n"
    "            m_k_shift = (\n"
    "                m_k_rot.reshape(m_k_rot.shape[0], kv_heads, head_dim)\n"
    "                + center_pre[None]\n"
    "            ).reshape(m_k_rot.shape)\n"
    "            m_q_hat = _dequantize_hif4(_dense_to_hif4(m_q_rot)).to(torch.float32)\n"
    "            m_k_hat = _dequantize_hif4(_dense_to_hif4(m_k_shift)).to(torch.float32)\n"
    "            m_output = _a2_attention_forward(\n"
    "                m_q_hat[None], m_k_hat[None], item[\"v_hat\"][None],\n"
    "                q_heads, kv_heads, head_dim,\n"
    "            )[0]\n"
    "            m_residual = m_output - item[\"reference\"]\n"
    "            m_d_output = (\n"
    "                2.0 * m_residual / float(m_residual.numel()) / item[\"mse_std\"]\n"
    "            )\n"
    "            m_d_qhat, m_d_khat = _m_attention_backward(\n"
    "                m_d_output[None], m_q_hat, m_k_hat, item[\"v_hat\"],\n"
    "                q_heads, kv_heads, head_dim,\n"
    "            )\n"
    "            m_tokens_q = item[\"q\"].shape[0]\n"
    "            m_tokens_k = item[\"k\"].shape[0]\n"
    "            m_per_group = q_heads // groups\n"
    "            m_q3g = item[\"q\"].reshape(m_tokens_q, groups, m_per_group, head_dim)\n"
    "            m_dq3g = m_d_qhat.reshape(m_tokens_q, groups, m_per_group, head_dim)\n"
    "            m_k3 = item[\"k\"].reshape(m_tokens_k, kv_heads, head_dim)\n"
    "            m_dk3 = m_d_khat.reshape(m_tokens_k, kv_heads, head_dim)\n"
    "            m_grad_rotation = torch.einsum(\"tghk,tghd->gkd\", m_q3g, m_dq3g)\n"
    "            m_grad_rotation = (\n"
    "                m_grad_rotation + torch.einsum(\"tgk,tgd->gkd\", m_k3, m_dk3)\n"
    "            )\n"
    "            m_grad_c = torch.einsum(\"kd,gkl->gdl\", base, m_grad_rotation)\n"
    "            m_grad_theta = m_grad_theta + _m_cayley_backward(m_grad_c, theta_pre)\n"
    "            m_grad_center = m_grad_center + m_dk3.sum(dim=0)\n"
    "        m_window_count = float(len(prepared))\n"
    "        m_grad_theta = m_grad_theta / m_window_count\n"
    "        m_grad_center = m_grad_center / m_window_count\n"
    "        m_c_pre, _m_right_pre = _m_cayley_pair(theta_pre)\n"
    "        m_grad_c_reg = (\n"
    "            2.0 * (m_c_pre - eye) * (_A2_REG_WEIGHT / float(groups * dim * dim))\n"
    "        )\n"
    "        m_grad_theta = m_grad_theta + _m_cayley_backward(m_grad_c_reg, theta_pre)\n"
    "        ah1_g_theta = m_grad_theta\n"
    "        ah1_g_center = m_grad_center\n"
)
source = replace_once(
    source,
    "    if identity_error > _A2_ORTHO_TOLERANCE:\n"
    "        raise RuntimeError(f\"A2 trained rotation failed orthogonality: {identity_error}\")\n"
    "    info = {\n",
    "    if identity_error > _A2_ORTHO_TOLERANCE:\n"
    "        raise RuntimeError(f\"A2 trained rotation failed orthogonality: {identity_error}\")\n"
    + mean_gradient_pass
    + "    info = {\n",
    "A-H1 mean-gradient direction pass",
)
source = replace_once(
    source,
    "        \"trainer\": \"manual+center\",\n"
    "    }\n",
    "        \"trainer\": \"manual+center\",\n"
    "        \"ah1_theta0\": theta_pre,\n"
    "        \"ah1_center0\": center_pre,\n"
    "        \"ah1_g_theta\": ah1_g_theta,\n"
    "        \"ah1_g_center\": ah1_g_center,\n"
    "    }\n",
    "A-H1 direction info fields",
)

# 3) Insert the A-H1 helpers before the final hif4_calibration_attention.
marker = "_V189_CALIBRATION_ATTENTION = hif4_calibration_attention"
function_start = source.index(
    "def hif4_calibration_attention(\n", source.index(marker)
)
ah1_block = AH1_CODE.read_text(encoding="utf-8").rstrip() + "\n\n\n"
source = source[:function_start] + ah1_block + source[function_start:]

# 4) Run the event search after the A2 gate; selection is applied by the
#    caller only on strict fold improvement.  An A-H1 failure keeps the
#    gate-selected parent state.
source = replace_once(
    source,
    "        states[\"q_state\"].update(audit)\n"
    "        states[\"k_state\"].update(audit)\n"
    "    except Exception:  # noqa: BLE001 - any failure degrades to exact R1\n",
    "        states[\"q_state\"].update(audit)\n"
    "        states[\"k_state\"].update(audit)\n"
    "        try:\n"
    "            ah1_audit, ah1_selected = _ah1_threshold_event_search(\n"
    "                calib_qkv_list,\n"
    "                windows,\n"
    "                states,\n"
    "                info,\n"
    "                q_num_heads,\n"
    "                kv_num_heads,\n"
    "                head_dim,\n"
    "                device,\n"
    "            )\n"
    "        except Exception as exc:  # noqa: BLE001 - keep the gate-selected parent state\n"
    "            ah1_audit = {\n"
    "                \"ah1_status\": f\"error:{type(exc).__name__}\",\n"
    "                \"ah1_attempted\": 0,\n"
    "                \"ah1_accepted\": 0,\n"
    "                \"ah1_accepted_event\": 0,\n"
    "            }\n"
    "            ah1_selected = None\n"
    "        if ah1_selected is not None:\n"
    "            ah1_rotation, ah1_center = ah1_selected\n"
    "            states[\"q_state\"][\"learned_rotation\"] = ah1_rotation\n"
    "            states[\"k_state\"][\"learned_rotation\"] = ah1_rotation.clone()\n"
    "            states[\"k_state\"][\"learned_center\"] = ah1_center\n"
    "        states[\"q_state\"].update(ah1_audit)\n"
    "        states[\"k_state\"].update(ah1_audit)\n"
    "    except Exception:  # noqa: BLE001 - any failure degrades to exact R1\n",
    "A-H1 calibration hook",
)

CANDIDATE.parent.mkdir(parents=True, exist_ok=True)
CANDIDATE.write_text(source, encoding="utf-8")
candidate_sha256 = hashlib.sha256(CANDIDATE.read_bytes()).hexdigest()
config = {
    "run_id": "attention-ah1-threshold-events",
    "version": "unassigned-until-material-output-change",
    "mechanism": "A-H1 quantization threshold event search along the window-normalized A2 direction",
    "parent": "solution.py",
    "parent_sha256": actual_parent_sha256,
    "candidate_sha256": candidate_sha256,
    "changes": [
        "trainer snapshots theta/center before the final Adam update (root arithmetic unchanged)",
        "window-mean direction gradient recomputed at the snapshot point",
        "analytic Cayley-path derivative and linearized HiF4 magnitude-midpoint events",
        "8 fixed event slots, true deployed-path fold selection, holdout record-only",
    ],
    "official_status": "unregistered/NA",
}
(HERE / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
print(json.dumps(config, indent=2))
