"""A-PG1, second attempt: target the field that is actually deployed.

The first attempt wrote `rotation_block`, which is dead (the state dump showed
`rotation=None, rotation_block=None` while the dynamic call passed block=8).
The live path is `solution.py:4226-4239`:

    if int(block_smooth_size) != 0:
        if attention_block_signs is not None:
            dense = _apply_attention_rotation(..., block_signs, int(block_smooth_size))
        else:
            dense = _block_hadamard_transform(dense, block_smooth_size, block_smooth_seed)

so the deployed rotation is `_apply_attention_rotation` driven by
`block_smooth_signs` (kv_heads, head_dim) and `block_smooth_size`.

Four byte-level hunks, each with a uniqueness assert and a head/tail check.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "solution.py"
PARENT = ROOT / "solutions" / "20260911_v250_linear-lem3-k6_scoreNA_timeNA" / "solution.py"


def lines(*rows: str) -> bytes:
    return ("\r\n".join(rows) + "\r\n").encode()


# H1: per-group block support in the rotation.  Int input behaves exactly as
# before, so every legacy state is untouched.
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

# H2: `block_smooth_size` may now be a per-group list, so the deployed branch
# must not coerce it with int().
OLD2 = lines(
    "    if int(block_smooth_size) != 0:",
    "        if attention_block_signs is not None:",
    "            if rotation_num_heads is None:",
    '                raise ValueError("Attention block smoothing requires head count")',
    '            block_signs = attention_block_signs.detach().to(device="cpu")',
    "            if block_signs.ndim != 2 or int(block_signs.shape[1]) <= 0:",
    '                raise ValueError("Attention block signs have invalid shape")',
    "            dense = _apply_attention_rotation(",
    "                dense,",
    "                int(rotation_num_heads),",
    "                int(block_signs.shape[1]),",
    "                block_signs,",
    "                int(block_smooth_size),",
    "            )",
)
NEW2 = lines(
    "    _bs_nonzero = (",
    "        any(int(_b) != 0 for _b in block_smooth_size)",
    "        if isinstance(block_smooth_size, (list, tuple))",
    "        else int(block_smooth_size) != 0",
    "    )",
    "    if _bs_nonzero:",
    "        if attention_block_signs is not None:",
    "            if rotation_num_heads is None:",
    '                raise ValueError("Attention block smoothing requires head count")',
    '            block_signs = attention_block_signs.detach().to(device="cpu")',
    "            if block_signs.ndim != 2 or int(block_signs.shape[1]) <= 0:",
    '                raise ValueError("Attention block signs have invalid shape")',
    "            dense = _apply_attention_rotation(",
    "                dense,",
    "                int(rotation_num_heads),",
    "                int(block_signs.shape[1]),",
    "                block_signs,",
    "                block_smooth_size,",
    "            )",
)

# H3: the grouped arm -- same forward pass, plus the per-KV-group causal MSE.
OLD3 = lines(
    "        causal_scores.append(float((out_c - ref_c).square().mean()))",
    "        safety_scores.append(float((out_n - ref_n).square().mean()))",
    "    return causal_scores, safety_scores",
)
NEW3 = lines(
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

# H4: track the per-group argmin in the C76.4 loop and write it into
# block_smooth_size / block_smooth_signs -- the fields the deployment reads.
OLD4 = lines(
    "            rotation_causal, rotation_safety = _attention_deployed_mse(",
    "                a1_q_pairs,",
    "                a1_k_pairs,",
    "                a1_v_hats,",
    '                a1_context["refs"],',
    "                rotation_q_state,",
    "                rotation_k_state,",
    "                q_num_heads,",
    "                kv_num_heads,",
    "                head_dim,",
    "            )",
)
NEW4 = lines(
    "            (",
    "                rotation_causal,",
    "                rotation_safety,",
    "                rotation_group,",
    "            ) = _attention_deployed_mse_grouped(",
    "                a1_q_pairs,",
    "                a1_k_pairs,",
    "                a1_v_hats,",
    '                a1_context["refs"],',
    "                rotation_q_state,",
    "                rotation_k_state,",
    "                q_num_heads,",
    "                kv_num_heads,",
    "                head_dim,",
    "            )",
)

OLD5 = lines(
    "                rotation_mean = sum(rotation_causal) / len(rotation_causal)",
    "                if rotation_mean < best_rotation_mean:",
)
NEW5 = lines(
    "                rotation_mean = sum(rotation_causal) / len(rotation_causal)",
    "                # A-PG1: the same forward pass already carries the error per KV",
    "                # group, so the argmin can be taken per group at no extra cost.",
    "                if pg_best_mse is None:",
    "                    pg_best_mse = rotation_group.clone()",
    "                    pg_best_row = [",
    "                        (signs[g].clone(), block) for g in range(kv_num_heads)",
    "                    ]",
    "                else:",
    "                    _better = rotation_group < pg_best_mse",
    "                    if bool(_better.any()):",
    "                        pg_best_mse = torch.where(",
    "                            _better, rotation_group, pg_best_mse",
    "                        )",
    "                        for g in range(kv_num_heads):",
    "                            if bool(_better[g]):",
    "                                pg_best_row[g] = (signs[g].clone(), block)",
    "                if rotation_mean < best_rotation_mean:",
)

OLD6 = lines(
    "                    best_rotation_states = (",
    "                        rotation_q_state,",
    "                        rotation_k_state,",
    "                    )",
    "        if best_rotation_states is not None:",
    "            q_state, k_state = best_rotation_states",
)
NEW6 = lines(
    "                    best_rotation_states = (",
    "                        rotation_q_state,",
    "                        rotation_k_state,",
    "                    )",
    "        if best_rotation_states is not None:",
    "            q_state, k_state = best_rotation_states",
    "        if pg_best_row is not None and len({b for _r, b in pg_best_row}) > 1:",
    "            # A-PG1: the per-group argmin picks different blocks for different",
    "            # KV groups, so the deployed block_smooth_size becomes a list and the",
    "            # deployed signs take each group's own row.  No new state field:",
    "            # block_smooth_signs is already (kv_heads, head_dim).",
    "            _pg_blocks = [int(b) for _r, b in pg_best_row]",
    "            _pg_signs = _cpu_state_tensor(",
    "                torch.stack([r for r, _b in pg_best_row], dim=0)",
    "            )",
    "            for _st in (q_state, k_state):",
    '                _st["block_smooth_size"] = _pg_blocks',
    '                _st["block_smooth_signs"] = _pg_signs',
)

OLD7 = lines(
    "        best_rotation_states = None",
    "        best_rotation_mean = base_rotation_mean",
)
NEW7 = lines(
    "        best_rotation_states = None",
    "        best_rotation_mean = base_rotation_mean",
    "        pg_best_mse = None",
    "        pg_best_row = None",
)


def replace_once(blob: bytes, old: bytes, new: bytes, tag: str) -> bytes:
    hits = blob.count(old)
    if hits != 1:
        raise SystemExit(f"{tag}: expected 1 occurrence, found {hits} -- refusing")
    k = blob.index(old)
    out = blob[:k] + new + blob[k + len(old):]
    if out[:k] != blob[:k] or out[k + len(new):] != blob[k + len(old):]:
        raise SystemExit(f"{tag}: head/tail identity FAILED")
    print(f"  {tag}: ok (pos {k})")
    return out


def main() -> int:
    src = PARENT.read_bytes()
    print(f"parent: {len(src)} B sha {hashlib.sha256(src).hexdigest()[:16]}")
    out = src
    for old, new, tag in (
        (OLD1, NEW1, "h1 per-group rotation"),
        (OLD2, NEW2, "h2 deployed branch"),
        (OLD3, NEW3, "h3 grouped arm"),
        (OLD4, NEW4, "h4 grouped call"),
        (OLD7, NEW7, "h5 trackers"),
        (OLD5, NEW5, "h6 argmin"),
        (OLD6, NEW6, "h7 write fields"),
    ):
        out = replace_once(out, old, new, tag)
    SRC.write_bytes(out)
    print(f"candidate: {len(out)} B sha {hashlib.sha256(out).hexdigest().upper()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
