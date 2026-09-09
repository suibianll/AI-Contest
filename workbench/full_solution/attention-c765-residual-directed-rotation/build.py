"""Build the A-C76.5 residual-directed C76.4 candidate from the v202 root."""

from pathlib import Path
import hashlib
import json


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
PARENT = ROOT / "solution.py"
CANDIDATE = HERE / "candidate" / "solution.py"
C765_CODE = HERE / "c765_code.py"
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
    "    except Exception as exc:  # noqa: BLE001 - propagate instead of masking\n"
    "        raise RuntimeError(\n"
    "            f\"A2 attention calibration failed: {type(exc).__name__}: {exc}\"\n"
    "        ) from exc\n"
    "    return states\n",
    "A2 calibration except propagation",
)

marker = "_V189_CALIBRATION_ATTENTION = hif4_calibration_attention"
function_start = source.index(
    "def hif4_calibration_attention(\n", source.index(marker)
)
c765_block = C765_CODE.read_text(encoding="utf-8").rstrip() + "\n\n\n"
source = source[:function_start] + c765_block + source[function_start:]

source = replace_once(
    source,
    "        base_rotation_mean = sum(base_causal) / max(len(base_causal), 1)\n"
    "        best_rotation_states = None\n"
    "        best_rotation_mean = base_rotation_mean\n"
    "        for block_size in _ATTN_ROTATION_BLOCKS:\n",
    "        base_rotation_mean = sum(base_causal) / max(len(base_causal), 1)\n"
    "        best_rotation_states = None\n"
    "        best_rotation_mean = base_rotation_mean\n"
    "        c765_coupling = _c765_qk_coupling(\n"
    "            a1_q_pairs,\n"
    "            a1_k_pairs,\n"
    "            a1_v_hats,\n"
    "            a1_context[\"refs\"],\n"
    "            q_state,\n"
    "            k_state,\n"
    "            q_num_heads,\n"
    "            kv_num_heads,\n"
    "            head_dim,\n"
    "        )\n"
    "        for block_size in _ATTN_ROTATION_BLOCKS:\n",
    "C76.5 coupling computation",
)
source = replace_once(
    source,
    "            for seed in _ATTN_ROTATION_SEEDS:\n"
    "                signs = _attention_rotation_signs(kv_num_heads, head_dim, int(seed))\n"
    "                rotation_q_state, rotation_k_state = _build_qk_states(\n",
    "            c765_candidates = [\n"
    "                _attention_rotation_signs(kv_num_heads, head_dim, int(seed))\n"
    "                for seed in _ATTN_ROTATION_SEEDS\n"
    "            ]\n"
    "            if block in _C765_BLOCKS and c765_coupling is not None:\n"
    "                c765_residual = _c765_signs_from_coupling(c765_coupling, block)\n"
    "                c765_duplicate = any(\n"
    "                    bool(torch.equal(c765_residual, candidate))\n"
    "                    for candidate in c765_candidates\n"
    "                )\n"
    "                print(\n"
    "                    f\"[C76.5] block={block} duplicate={int(c765_duplicate)} \"\n"
    "                    f\"residual_sum={float(c765_residual.sum()):.1f}\",\n"
    "                    flush=True,\n"
    "                )\n"
    "                if not c765_duplicate:\n"
    "                    c765_candidates.append(c765_residual)\n"
    "            for signs in c765_candidates:\n"
    "                rotation_q_state, rotation_k_state = _build_qk_states(\n",
    "C76.5 candidate injection",
)

CANDIDATE.parent.mkdir(parents=True, exist_ok=True)
CANDIDATE.write_text(source, encoding="utf-8")
candidate_sha256 = hashlib.sha256(CANDIDATE.read_bytes()).hexdigest()
config = {
    "run_id": "attention-c765-residual-directed-rotation",
    "version": "unassigned-until-material-output-change",
    "mechanism": "A-C76.5 residual-directed C76.4 orthogonal candidate",
    "parent": "solution.py",
    "parent_sha256": actual_parent_sha256,
    "candidate_sha256": candidate_sha256,
    "changes": [
        "case-equal Q/K right-transform coupling at the C76.4 parent state",
        "one dominant-eigenvector sign candidate per block size B in {16, 32}",
        "candidates dropped into the existing complete deployed-MSE selection, C76.4 unchanged",
        "remove two encoder-path broad excepts and propagate the two calibration broad excepts",
    ],
    "official_status": "unregistered/NA",
}
(HERE / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
print(json.dumps(config, indent=2))
