"""Build the A-H1R parent-anchored threshold-event candidate from the v202 root."""

from pathlib import Path
import hashlib
import json


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
PARENT = ROOT / "solution.py"
CANDIDATE = HERE / "candidate" / "solution.py"
AH1R_CODE = HERE / "ah1r_code.py"
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

# 1) Remove the two encoder-path broad excepts that hide learned rotation/center.
source = replace_once(
    source,
    "    if learned_rotation is not None and learned_rotation_num_heads is not None:\n"
    "        try:\n"
    "            dense = _a2_apply_group_rotation(\n"
    "                dense, int(learned_rotation_num_heads), learned_rotation\n"
    "            )\n"
    "        except Exception:  # noqa: BLE001 - degrade to the unrotated legal path\n"
    "            pass\n",
    "    if learned_rotation is not None and learned_rotation_num_heads is not None:\n"
    "        dense = _a2_apply_group_rotation(\n"
    "            dense, int(learned_rotation_num_heads), learned_rotation\n"
    "        )\n",
    "encoder learned_rotation except removal",
)
source = replace_once(
    source,
    "    if learned_center is not None and learned_rotation_num_heads is not None:\n"
    "        try:\n"
    "            heads = int(learned_rotation_num_heads)\n"
    "            head_dim_c = int(dense.shape[-1]) // heads\n"
    "            lead = dense.shape[:-1]\n"
    "            dense = (\n"
    "                dense.reshape(*lead, heads, head_dim_c)\n"
    "                + learned_center.to(device=dense.device, dtype=torch.float32).reshape(\n"
    "                    *([1] * len(lead)), heads, head_dim_c\n"
    "                )\n"
    "            ).reshape(dense.shape)\n"
    "        except Exception:  # noqa: BLE001 - degrade to the unshifted legal path\n"
    "            pass\n",
    "    if learned_center is not None and learned_rotation_num_heads is not None:\n"
    "        heads = int(learned_rotation_num_heads)\n"
    "        head_dim_c = int(dense.shape[-1]) // heads\n"
    "        lead = dense.shape[:-1]\n"
    "        dense = (\n"
    "            dense.reshape(*lead, heads, head_dim_c)\n"
    "            + learned_center.to(device=dense.device, dtype=torch.float32).reshape(\n"
    "                *([1] * len(lead)), heads, head_dim_c\n"
    "            )\n"
    "        ).reshape(dense.shape)\n",
    "encoder learned_center except removal",
)

# 2) Propagate the two calibration-path broad excepts instead of masking them.
source = replace_once(
    source,
    "    except Exception:  # noqa: BLE001 - unreachable here; kept for safety\n"
    "        return {\"q_state\": {}, \"k_state\": {}, \"v_state\": {}}\n",
    "    except Exception as exc:  # noqa: BLE001 - propagate instead of masking\n"
    "        raise RuntimeError(\n"
    "            f\"v189 attention calibration failed: {type(exc).__name__}: {exc}\"\n"
    "        ) from exc\n",
    "v189 calibration except propagation",
)

# 3) Insert the A-H1R helpers before the final hif4_calibration_attention.
marker = "_V189_CALIBRATION_ATTENTION = hif4_calibration_attention"
function_start = source.index(
    "def hif4_calibration_attention(\n", source.index(marker)
)
ah1r_block = AH1R_CODE.read_text(encoding="utf-8").rstrip() + "\n\n\n"
source = source[:function_start] + ah1r_block + source[function_start:]

# 4) Run the parent-anchored event search after the A2 gate, assert the deployed
#    path really applies the stored transform, and propagate failures.
source = replace_once(
    source,
    "        states[\"q_state\"].update(audit)\n"
    "        states[\"k_state\"].update(audit)\n"
    "    except Exception:  # noqa: BLE001 - any failure degrades to exact R1\n"
    "        states[\"q_state\"].pop(\"learned_rotation\", None)\n"
    "        states[\"k_state\"].pop(\"learned_rotation\", None)\n"
    "        states[\"k_state\"].pop(\"learned_center\", None)\n"
    "        states[\"q_state\"][\"a2_arm\"] = \"fallback\"\n"
    "        states[\"k_state\"][\"a2_arm\"] = \"fallback\"\n"
    "    return states\n",
    "        states[\"q_state\"].update(audit)\n"
    "        states[\"k_state\"].update(audit)\n"
    "        if states[\"q_state\"].get(\"a2_arm\") == \"fallback\":\n"
    "            raise RuntimeError(\n"
    "                \"A2 fallback state must not reach the A-H1R search\"\n"
    "            )\n"
    "        ah1r_audit, ah1r_selected = _ah1r_threshold_event_search(\n"
    "            calib_qkv_list,\n"
    "            windows,\n"
    "            states,\n"
    "            q_num_heads,\n"
    "            kv_num_heads,\n"
    "            head_dim,\n"
    "            device,\n"
    "        )\n"
    "        if ah1r_selected is not None:\n"
    "            ah1r_rotation, ah1r_center = ah1r_selected\n"
    "            states[\"q_state\"][\"learned_rotation\"] = ah1r_rotation\n"
    "            states[\"k_state\"][\"learned_rotation\"] = ah1r_rotation.clone()\n"
    "            states[\"k_state\"][\"learned_center\"] = ah1r_center\n"
    "        states[\"q_state\"].update(ah1r_audit)\n"
    "        states[\"k_state\"].update(ah1r_audit)\n"
    "        _ah1r_assert_dynamic_applies(\n"
    "            calib_qkv_list, states, q_num_heads, kv_num_heads, head_dim\n"
    "        )\n"
    "    except Exception as exc:  # noqa: BLE001 - propagate instead of masking\n"
    "        raise RuntimeError(\n"
    "            f\"A2/A-H1R attention calibration failed: {type(exc).__name__}: {exc}\"\n"
    "        ) from exc\n"
    "    return states\n",
    "A-H1R calibration hook",
)

CANDIDATE.parent.mkdir(parents=True, exist_ok=True)
CANDIDATE.write_text(source, encoding="utf-8")
candidate_sha256 = hashlib.sha256(CANDIDATE.read_bytes()).hexdigest()
config = {
    "run_id": "attention-ah1-parent-anchored",
    "version": "unassigned-until-material-output-change",
    "mechanism": "A-H1R deployed-parent-anchored quantization threshold event search",
    "parent": "solution.py",
    "parent_sha256": actual_parent_sha256,
    "candidate_sha256": candidate_sha256,
    "changes": [
        "anchor events at the gate-selected deployed parent rotation/center (identity/zero arm supported)",
        "recompute the fold-mean output gradient in the deployed pre-rotation coordinates",
        "tangent direction S = skew(R_parent^T G_R), path R(t) = R_parent @ cayley(t S), center fixed",
        "remove two encoder-path broad excepts and propagate the two calibration broad excepts",
        "assert a2_arm != fallback and that the dynamic path really applies the stored transform",
        "8 fixed event slots, true deployed-path fold selection, holdout record-only",
    ],
    "official_status": "unregistered/NA",
}
(HERE / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
print(json.dumps(config, indent=2))
