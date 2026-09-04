"""One-shot v184 dual-window refactor edit (2026-09-04)."""
import re
from pathlib import Path

P = Path(r"solutions\20260904_v184_attn-plus4-gate_scoreNA_timeNA\solution.py")
src = P.read_text(encoding="utf-8")

# 1. module-level active window for _attention_candidate_metrics
old = "_DYNAMIC_OFFSETS = (-1, 1, 2, 3)"
new = (
    "_DYNAMIC_OFFSETS = (-1, 1, 2, 3)\n"
    "# v184: mutable active window read by _attention_candidate_metrics so the\n"
    "# dual-window wrapper can rerun the whole calibration under a wider\n"
    "# scale-search window without threading a parameter through every call.\n"
    "_ACTIVE_ATTN_OFFSET_WINDOW = _DYNAMIC_OFFSETS"
)
assert src.count(old) == 1
src = src.replace(old, new)

# 2. candidate metrics reads the active window (2 sites inside that function only)
fn_start = src.index("def _attention_candidate_metrics(")
fn_end = src.index("\ndef ", fn_start)
seg = src[fn_start:fn_end]
old = "                    search_offsets=_DYNAMIC_OFFSETS,"
assert seg.count(old) == 2, seg.count(old)
seg = seg.replace(old, "                    search_offsets=_ACTIVE_ATTN_OFFSET_WINDOW,")
src = src[:fn_start] + seg + src[fn_end:]

# 3. core entry sets the global from its parameter
old = '''    if not isinstance(calib_qkv_list, list) or not calib_qkv_list:
        raise ValueError("calib_qkv_list must be a non-empty list")
    if q_num_heads <= 0 or kv_num_heads <= 0 or head_dim <= 0:
        raise ValueError("head counts and head_dim must be positive")
    if q_num_heads % kv_num_heads != 0:
        raise ValueError("q_num_heads must be divisible by kv_num_heads")
    q_channels = q_num_heads * head_dim'''
new = '''    global _ACTIVE_ATTN_OFFSET_WINDOW
    _ACTIVE_ATTN_OFFSET_WINDOW = tuple(offset_window)
    if not isinstance(calib_qkv_list, list) or not calib_qkv_list:
        raise ValueError("calib_qkv_list must be a non-empty list")
    if q_num_heads <= 0 or kv_num_heads <= 0 or head_dim <= 0:
        raise ValueError("head counts and head_dim must be positive")
    if q_num_heads % kv_num_heads != 0:
        raise ValueError("q_num_heads must be divisible by kv_num_heads")
    q_channels = q_num_heads * head_dim'''
assert src.count(old) == 1
src = src.replace(old, new)

# 4. v_state multiline offsets -> offset_window
old = '''            "offsets": torch.tensor(
                _DYNAMIC_OFFSETS, dtype=torch.int8, device="cpu"
            ),'''
assert src.count(old) == 1
src = src.replace(
    old,
    '''            "offsets": torch.tensor(
                offset_window, dtype=torch.int8, device="cpu"
            ),''',
)

# 5. q/k state single-line offsets -> offset_window (2 sites)
old = '"offsets": torch.tensor(_DYNAMIC_OFFSETS, dtype=torch.int8, device="cpu"),'
assert src.count(old) == 2, src.count(old)
src = src.replace(
    old,
    '"offsets": torch.tensor(offset_window, dtype=torch.int8, device="cpu"),',
)

# 6. remove the snapshot block (from its comment to the A1 block start)
snap_start = "    # v184: snapshot the pre-gamma Q/K/V states so the +4-window candidate"
snap_end = "    if _ATTN_LOGIT_GAIN and a1_q_pairs and a1_k_pairs:"
i = src.index(snap_start)
j = src.index(snap_end, i)
src = src[:i] + src[j:]

# 7. remove the consistency gate block (from its comment to the core return)
gate_start = "    # v184: per-layer gated +4 scale code with fit-deploy consistency."
ret = '    return {"q_state": q_state, "k_state": k_state, "v_state": v_state}'
i = src.index(gate_start)
j = src.index(ret, i) + len(ret)
core_tail = '''    result = {"q_state": q_state, "k_state": k_state, "v_state": v_state}
    if return_eval_context:
        eval_ctx = {
            "q_pairs": a1_q_pairs,
            "k_pairs": a1_k_pairs,
            "v_pairs": a1_v_pairs,
            "refs": (
                a1_context["refs"] if a1_context is not None else None
            ),
        }
        return result, eval_ctx
    return result'''
src = src[:i] + core_tail + src[j:]

# 8. append the dual-window wrapper after the core function
wrapper = '''


_ATTN_PLUS4_WINDOW = (*_DYNAMIC_OFFSETS, 4)


@torch.no_grad()
def hif4_calibration_attention(
    calib_qkv_list: list,
    q_num_heads: int,
    kv_num_heads: int,
    head_dim: int,
) -> dict[str, Any]:
    """v184 dual-window wrapper: run the full calibration once under the
    standard 4-code window (bit-identical to v182 when the gate rejects) and
    once under the +4 5-code window; keep whichever branch has the better
    deployed calibration MSE under the same no-degradation gate as the
    A1/rotation paths (safety_tolerance=0.0)."""

    base_states = _calibrate_attention_core(
        calib_qkv_list, q_num_heads, kv_num_heads, head_dim, _DYNAMIC_OFFSETS
    )
    if not _ATTN_PLUS4_OFFSET_GATE:
        return base_states
    cand_states, eval_ctx = _calibrate_attention_core(
        calib_qkv_list,
        q_num_heads,
        kv_num_heads,
        head_dim,
        _ATTN_PLUS4_WINDOW,
        return_eval_context=True,
    )
    refs = eval_ctx["refs"]
    q_pairs = eval_ctx["q_pairs"]
    k_pairs = eval_ctx["k_pairs"]
    v_pairs = eval_ctx["v_pairs"]
    if refs is None or not q_pairs or not k_pairs or not v_pairs:
        return base_states

    def _branch_v_hats(v_state):
        return [
            _dequantize_hif4(
                hif4_dynamic_quantize_v(
                    v_quant, v_scale, kv_num_heads, head_dim, v_state
                )
            ).to(torch.float32)
            for v_quant, v_scale in v_pairs
        ]

    base_causal, base_safety = _attention_deployed_mse(
        q_pairs,
        k_pairs,
        _branch_v_hats(base_states["v_state"]),
        refs,
        base_states["q_state"],
        base_states["k_state"],
        q_num_heads,
        kv_num_heads,
        head_dim,
    )
    cand_causal, cand_safety = _attention_deployed_mse(
        q_pairs,
        k_pairs,
        _branch_v_hats(cand_states["v_state"]),
        refs,
        cand_states["q_state"],
        cand_states["k_state"],
        q_num_heads,
        kv_num_heads,
        head_dim,
    )
    base_mean = sum(base_causal) / len(base_causal)
    cand_mean = sum(cand_causal) / len(cand_causal)
    if _a1_gate_passes(
        cand_causal,
        cand_safety,
        base_causal,
        base_safety,
        safety_tolerance=0.0,
    ):
        print(
            f"[v184] +4 window ACCEPTED base={base_mean:.6f} cand={cand_mean:.6f}",
            flush=True,
        )
        return cand_states
    print(
        f"[v184] +4 window rejected base={base_mean:.6f} cand={cand_mean:.6f}",
        flush=True,
    )
    return base_states
'''
anchor = "def _check_attention_state("
i = src.index(anchor)
# insert before the blank lines preceding _check_attention_state
head = src[: i - 2]  # strip the two newlines before def
src = head + wrapper + "\n\n" + src[i - 2:]

P.write_text(src, encoding="utf-8", newline="\n")
print("edit OK")
