"""Build the A-PG1 candidate: per-KV-group rotation selection in the C76.4 search.

The score is an elementwise MSE and every output element belongs to exactly one
head, so the error is additive across heads and therefore across KV groups.  The
selection is currently taken once per layer; taking it per group costs no extra
forward pass, only a finer argmin.  Measured headroom: 4 of 6 layers have a
different best block per group, mean 1.32% MSE cut (APG1-result.md).

Byte-level hunks, each with a uniqueness assert and a head/tail identity check.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "solution.py"
PARENT = ROOT / "solutions" / "20260911_v250_linear-lem3-k6_scoreNA_timeNA" / "solution.py"


def lines(*rows: str) -> bytes:
    return ("\r\n".join(rows) + "\r\n").encode()


# --------------------------------------------------------------------------
# HUNK 1 -- per-group block support in the rotation itself.  Identical to the
# previous behaviour whenever block_size is an int, which is the only form any
# existing state carries, so legacy states keep working untouched.
# --------------------------------------------------------------------------
OLD1 = lines(
    "    requested_block = int(",
    "        _ATTN_H64_BLOCK if block_size is None else block_size",
    "    )",
    "    block = requested_block if requested_block > 0 and head_dim % requested_block == 0 else head_dim",
)

NEW1 = lines(
    "    if isinstance(block_size, (list, tuple)) or torch.is_tensor(block_size):",
    "        # A-PG1: one Hadamard block per KV group.  The score is an elementwise",
    "        # MSE and every output element belongs to exactly one head, so the error",
    "        # is additive across heads and therefore across KV groups; choosing the",
    "        # block per group costs no extra forward pass, only a finer argmin.",
    "        per_group = [int(b) for b in block_size]",
    "        if len(per_group) != kv_num_heads:",
    "            per_group = [per_group[0] if per_group else _ATTN_H64_BLOCK] * kv_num_heads",
    "        xg = x.reshape(*x.shape[:-1], kv_num_heads, max(group_size, 1), head_dim)",
    "        out = []",
    "        for g in range(kv_num_heads):",
    "            blk = per_group[g]",
    "            if blk <= 0 or head_dim % blk != 0:",
    "                blk = head_dim",
    "            seg = xg[..., g, :, :]",
    "            h = _hadamard_matrix_unchecked(blk, dense.device, dense.dtype)",
    "            seg = seg.reshape(*seg.shape[:-1], head_dim // blk, blk)",
    "            out.append(torch.matmul(seg, h).reshape(*seg.shape[:-2], head_dim))",
    "        return torch.stack(out, dim=-2).reshape(*dense.shape)",
    "    requested_block = int(",
    "        _ATTN_H64_BLOCK if block_size is None else block_size",
    "    )",
    "    block = requested_block if requested_block > 0 and head_dim % requested_block == 0 else head_dim",
)

# --------------------------------------------------------------------------
# HUNK 2 -- the grouped arm.  The same computation as _attention_deployed_mse
# plus the per-KV-group causal MSE, taken from the SAME forward pass.
# --------------------------------------------------------------------------
OLD2 = lines(
    "        causal_scores.append(float((out_c - ref_c).square().mean()))",
    "        safety_scores.append(float((out_n - ref_n).square().mean()))",
    "    return causal_scores, safety_scores",
)

NEW2 = lines(
    "        causal_scores.append(float((out_c - ref_c).square().mean()))",
    "        safety_scores.append(float((out_n - ref_n).square().mean()))",
    "    return causal_scores, safety_scores",
    "",
    "",
    "def _attention_deployed_mse_grouped(",
    "    q_pairs: list,",
    "    k_pairs: list,",
    "    v_hats: list,",
    "    refs: list,",
    "    q_state: dict,",
    "    k_state: dict,",
    "    q_num_heads: int,",
    "    kv_num_heads: int,",
    "    head_dim: int,",
    ") -> tuple:",
    '    """`_attention_deployed_mse` plus the exact per-KV-group causal MSE.',
    "",
    "    A-PG1: the per-group numbers are free -- they come from the same forward",
    "    pass, because the score is an elementwise MSE and each output element",
    "    belongs to exactly one head.",
    '    """',
    "",
    "    causal_scores: list = []",
    "    safety_scores: list = []",
    "    group_scores: list = []",
    "    for (q_quant, q_scale), (k_quant, k_scale), v_hat, (ref_c, ref_n) in zip(",
    "        q_pairs, k_pairs, v_hats, refs",
    "    ):",
    "        q_hat = _dequantize_hif4(",
    "            hif4_dynamic_quantize_q(",
    "                q_quant, q_scale, q_num_heads, head_dim, q_state",
    "            )",
    "        ).to(torch.float32)",
    "        k_hat = _dequantize_hif4(",
    "            hif4_dynamic_quantize_k(",
    "                k_quant, k_scale, kv_num_heads, head_dim, k_state",
    "            )",
    "        ).to(torch.float32)",
    "        out_c = _attention_forward(",
    "            q_hat, k_hat, v_hat, q_num_heads, kv_num_heads, head_dim, True",
    "        )",
    "        out_n = _attention_forward(",
    "            q_hat, k_hat, v_hat, q_num_heads, kv_num_heads, head_dim, False",
    "        )",
    "        causal_scores.append(float((out_c - ref_c).square().mean()))",
    "        safety_scores.append(float((out_n - ref_n).square().mean()))",
    "        flat = (out_c - ref_c).square().reshape(",
    "            -1, q_num_heads, head_dim",
    "        ).mean(dim=(0, 2))",
    "        group_scores.append(",
    "            flat.reshape(kv_num_heads, q_num_heads // kv_num_heads).mean(dim=1)",
    "        )",
    "    return causal_scores, safety_scores, torch.stack(group_scores).mean(dim=0)",
)



# --------------------------------------------------------------------------
# HUNK 3 -- track the per-group argmin alongside the per-layer one.
# --------------------------------------------------------------------------
OLD3 = lines(
    "        best_rotation_states = None",
    "        best_rotation_mean = base_rotation_mean",
)
NEW3 = lines(
    "        best_rotation_states = None",
    "        best_rotation_mean = base_rotation_mean",
    "        # A-PG1: the per-KV-group argmin, kept alongside the per-layer one.",
    "        group_best_mse = None",
    "        group_best_row = None",
)

# --------------------------------------------------------------------------
# HUNK 4 -- use the grouped arm and assemble a per-group state.
# --------------------------------------------------------------------------
OLD4 = lines(
    "                rotation_causal, rotation_safety = _attention_deployed_mse(",
    "                    a1_q_pairs,",
    "                    a1_k_pairs,",
    "                    a1_v_hats,",
    '                    a1_context["refs"],',
    "                    rotation_q_state,",
    "                    rotation_k_state,",
    "                    q_num_heads,",
    "                    kv_num_heads,",
    "                    head_dim,",
    "                )",
)
NEW4 = lines(
    "                (",
    "                    rotation_causal,",
    "                    rotation_safety,",
    "                    rotation_group,",
    "                ) = _attention_deployed_mse_grouped(",
    "                    a1_q_pairs,",
    "                    a1_k_pairs,",
    "                    a1_v_hats,",
    '                    a1_context["refs"],',
    "                    rotation_q_state,",
    "                    rotation_k_state,",
    "                    q_num_heads,",
    "                    kv_num_heads,",
    "                    head_dim,",
    "                )",
)

OLD5 = lines(
    "                if rotation_mean < best_rotation_mean:",
    "                    best_rotation_mean = rotation_mean",
    "                    best_rotation_states = (",
    "                        rotation_q_state,",
    "                        rotation_k_state,",
    "                    )",
    "        if best_rotation_states is not None:",
    "            q_state, k_state = best_rotation_states",
)
NEW5 = lines(
    "                if rotation_mean < best_rotation_mean:",
    "                    best_rotation_mean = rotation_mean",
    "                    best_rotation_states = (",
    "                        rotation_q_state,",
    "                        rotation_k_state,",
    "                    )",
    "                # A-PG1: the same forward pass already gives the error per KV",
    "                # group (the score is additive across heads), so the argmin can",
    "                # be taken per group at no extra cost.",
    "                if group_best_mse is None:",
    "                    group_best_mse = rotation_group.clone()",
    "                    group_best_row = [",
    "                        (signs[g].clone(), block) for g in range(kv_num_heads)",
    "                    ]",
    "                else:",
    "                    better = rotation_group < group_best_mse",
    "                    if bool(better.any()):",
    "                        group_best_mse = torch.where(",
    "                            better, rotation_group, group_best_mse",
    "                        )",
    "                        for g in range(kv_num_heads):",
    "                            if bool(better[g]):",
    "                                group_best_row[g] = (signs[g].clone(), block)",
    "        if group_best_row is not None:",
    "            # A-PG1: assemble one state whose rotation row and block come from",
    "            # each group's own argmin.  The score is additive across groups, so",
    "            # this is weakly better than the single per-layer choice, and it",
    "            # costs nothing extra -- the same candidates were already scored.",
    "            combined_signs = torch.stack([r for r, _b in group_best_row], dim=0)",
    "            q_state, k_state = _build_qk_states(",
    "                final_d,",
    "                int(final_center),",
    "                final_q_perm,",
    "                final_k_perm,",
    "                int(q_state.get(\"block_smooth_size\", 0)),",
    "                int(q_state.get(\"block_smooth_seed\", 0)),",
    "                rotation=combined_signs,",
    "                rotation_block=[b for _r, b in group_best_row],",
    "            )",
    "        elif best_rotation_states is not None:",
    "            q_state, k_state = best_rotation_states",
)

# --------------------------------------------------------------------------
# HUNK 6 -- the state may now carry a per-group block list, so the int()
# coercion has to accept both forms.  An int (every legacy state) is unchanged.
# --------------------------------------------------------------------------
OLD6 = lines(
    '            q_state["rotation_block"] = int(',
    "                _ATTN_H64_BLOCK if rotation_block is None else rotation_block",
    "            )",
    '            k_state["rotation_block"] = int(',
    "                _ATTN_H64_BLOCK if rotation_block is None else rotation_block",
    "            )",
)
NEW6 = lines(
    "            _block_value = (",
    "                _ATTN_H64_BLOCK if rotation_block is None else rotation_block",
    "            )",
    "            if isinstance(_block_value, (list, tuple)):",
    "                _block_value = [int(_b) for _b in _block_value]",
    "            else:",
    "                _block_value = int(_block_value)",
    '            q_state["rotation_block"] = _block_value',
    '            k_state["rotation_block"] = _block_value',
)

def replace_once(blob: bytes, old: bytes, new: bytes, tag: str) -> bytes:
    hits = blob.count(old)
    if hits != 1:
        raise SystemExit(f"{tag}: expected 1 occurrence, found {hits} -- refusing")
    k = blob.index(old)
    out = blob[:k] + new + blob[k + len(old):]
    if out[:k] != blob[:k] or out[k + len(new):] != blob[k + len(old):]:
        raise SystemExit(f"{tag}: head/tail identity FAILED")
    print(f"  {tag}: ok (pos {k}, {len(old)} -> {len(new)} B)")
    return out


def main() -> int:
    src = PARENT.read_bytes()
    print(f"parent: {len(src)} B sha {hashlib.sha256(src).hexdigest()[:16]}")
    out = replace_once(src, OLD1, NEW1, "hunk1 per-group rotation")
    out = replace_once(out, OLD2, NEW2, "hunk2 grouped arm")
    out = replace_once(out, OLD3, NEW3, "hunk3 trackers")
    out = replace_once(out, OLD4, NEW4, "hunk4 grouped call")
    out = replace_once(out, OLD5, NEW5, "hunk5 assemble state")
    out = replace_once(out, OLD6, NEW6, "hunk6 block coercion")
    SRC.write_bytes(out)
    print(f"candidate: {len(out)} B sha {hashlib.sha256(out).hexdigest().upper()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
