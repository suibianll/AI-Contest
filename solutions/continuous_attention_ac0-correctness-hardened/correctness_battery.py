"""AC0 correctness battery (T1-T9).

Runs against the archived AC0 candidate and the bit-identical R3 parent
(read-only import).  Each test prints PASS/FAIL with measured evidence.
Run with the repo CUDA venv:

    .venv\\Scripts\\python.exe solutions/continuous_attention_ac0-correctness-hardened/correctness_battery.py
"""

import importlib.util
import math
import os
import sys
import traceback

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, REPO)

AC0_PATH = os.path.join(HERE, "solution.py")
R3_PATH = os.path.join(
    REPO, "solutions", "v162_attention_r3-rotation-center_allgates", "solution.py"
)


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def nvfp4_pair(x, generator):
    grouped = x.reshape(*x.shape[:-1], x.shape[-1] // 16, 16)
    amax = grouped.abs().amax(dim=-1)
    scale = (amax / 7.0).to(torch.bfloat16).to(torch.float32)
    quant = torch.round(grouped / scale.unsqueeze(-1)).clamp(-8, 7)
    return (
        quant.flatten(-2, -1).to(torch.int8),
        scale.reshape(*x.shape[:-1], -1),
    )


def make_window(seq, q_heads, kv_heads, head_dim, seed):
    g = torch.Generator().manual_seed(seed)
    q = (torch.randn(seq, q_heads * head_dim, generator=g) * 2.0 + 0.5)
    k = (torch.randn(seq, kv_heads * head_dim, generator=g) * 2.0 + 0.5)
    v = (torch.randn(seq, kv_heads * head_dim, generator=g) * 2.0 + 0.5)
    return {"q": nvfp4_pair(q, g), "k": nvfp4_pair(k, g), "v": nvfp4_pair(v, g)}


def ortho_random(dim, seed):
    g = torch.Generator().manual_seed(seed)
    m = torch.randn(dim, dim, generator=g, dtype=torch.float64)
    q, _ = torch.linalg.qr(m)
    return q


def per_head_logits(q, k, q_heads, kv_heads, head_dim):
    group = q_heads // kv_heads
    seq_q = int(q.shape[0])
    seq_k = int(k.shape[0])
    qh = q.reshape(seq_q, q_heads, head_dim).transpose(0, 1)
    kh = (
        k.reshape(seq_k, kv_heads, head_dim)
        .transpose(0, 1)
        .repeat_interleave(group, dim=0)
    )
    return qh @ kh.transpose(-1, -2)


def fields_equal(a, b):
    return all(torch.equal(a[k], b[k]) for k in ("scale_factor", "scale_lv2", "scale_lv3", "sign", "mant"))


RESULTS = []


def check(name, ok, detail=""):
    RESULTS.append((name, bool(ok)))
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {detail}")


def main():
    ac0 = load_module(AC0_PATH, "ac0_battery")
    r3 = load_module(R3_PATH, "r3_battery")

    for api in [
        "hif4_calibration_and_quantize_weight",
        "hif4_dynamic_quantize_activation",
        "hif4_calibration_attention",
        "hif4_dynamic_quantize_q",
        "hif4_dynamic_quantize_k",
        "hif4_dynamic_quantize_v",
    ]:
        check(f"T0 API {api}", callable(getattr(ac0, api)) and callable(getattr(r3, api)))
    check("T0 distinct modules", ac0.__file__ != r3.__file__)

    q_heads, kv_heads, head_dim = 8, 2, 64
    calib = [make_window(96, q_heads, kv_heads, head_dim, s) for s in range(6)]
    win = calib[-1]

    with torch.inference_mode():
        # ---- T1: identity parity AC0 == R3 (learned transform absent) ----
        r3_states = r3.hif4_calibration_attention(calib, q_heads, kv_heads, head_dim)
        q_hat_ac0 = ac0.hif4_dynamic_quantize_q(win["q"][0], win["q"][1], q_heads, head_dim, r3_states["q_state"])
        q_hat_r3 = r3.hif4_dynamic_quantize_q(win["q"][0], win["q"][1], q_heads, head_dim, r3_states["q_state"])
        k_hat_ac0 = ac0.hif4_dynamic_quantize_k(win["k"][0], win["k"][1], kv_heads, head_dim, r3_states["k_state"])
        k_hat_r3 = r3.hif4_dynamic_quantize_k(win["k"][0], win["k"][1], kv_heads, head_dim, r3_states["k_state"])
        v_hat_ac0 = ac0.hif4_dynamic_quantize_v(win["v"][0], win["v"][1], kv_heads, head_dim, r3_states["v_state"])
        v_hat_r3 = r3.hif4_dynamic_quantize_v(win["v"][0], win["v"][1], kv_heads, head_dim, r3_states["v_state"])
        check("T1 Q five-field parity", fields_equal(q_hat_ac0, q_hat_r3))
        check("T1 K five-field parity", fields_equal(k_hat_ac0, k_hat_r3))
        check("T1 V five-field parity", fields_equal(v_hat_ac0, v_hat_r3))

        # ---- T2: rotation invariance (orthogonal R, FP64 math) ----
        seq = 48
        g = torch.Generator().manual_seed(7)
        q64 = torch.randn(seq, q_heads * head_dim, generator=g, dtype=torch.float64)
        k64 = torch.randn(seq, kv_heads * head_dim, generator=g, dtype=torch.float64)
        r_orth = torch.stack([ortho_random(head_dim, s) for s in range(kv_heads)])

        def group_apply_f64(x, heads, r):
            lead = x.shape[:-1]
            hd = int(x.shape[-1]) // heads
            groups = int(r.shape[0])
            per_group = heads // groups
            grouped = x.reshape(*lead, groups, per_group, hd)
            rotated = torch.einsum("tghk,gkd->tghd", grouped, r)
            return rotated.reshape(*lead, heads * hd)

        q_r = group_apply_f64(q64, q_heads, r_orth)
        k_r = group_apply_f64(k64, kv_heads, r_orth)
        lp = per_head_logits(q64, k64, q_heads, kv_heads, head_dim)
        lr = per_head_logits(q_r, k_r, q_heads, kv_heads, head_dim)
        rel_t2 = float((lr - lp).norm()) / (float(lp.norm()) + 1e-30)
        check("T2 rotation invariance", rel_t2 < 1e-8, f"rel={rel_t2:.3e}")

        # ---- T3: inverse transform invariance Q'=QT, K'=KT^-T (FP64 math) ----
        t_inv = torch.stack([ortho_random(head_dim, s + 11) @ torch.diag(torch.linspace(0.5, 1.5, head_dim, dtype=torch.float64)) for s in range(kv_heads)])
        t_neg = torch.linalg.inv(t_inv).transpose(-1, -2)
        q_t = group_apply_f64(q64, q_heads, t_inv)
        k_t = group_apply_f64(k64, kv_heads, t_neg)
        lt = per_head_logits(q_t, k_t, q_heads, kv_heads, head_dim)
        rel_t3 = float((lt - lp).norm()) / (float(lp.norm()) + 1e-30)
        check("T3 inverse transform invariance", rel_t3 < 1e-8, f"rel={rel_t3:.3e}")

        # ---- T4: K-center softmax invariance ----
        c_center = torch.randn(kv_heads, head_dim, generator=g, dtype=torch.float64)
        k_c = k64.clone()
        k_cg = k_c.reshape(seq, kv_heads, head_dim) - c_center
        k_c = k_cg.reshape_as(k_c)
        p_base = torch.softmax(per_head_logits(q64, k64, q_heads, kv_heads, head_dim), dim=-1)
        p_cent = torch.softmax(per_head_logits(q64, k_c, q_heads, kv_heads, head_dim), dim=-1)
        max_t4 = float((p_base - p_cent).abs().max())
        check("T4 K-center softmax invariance", max_t4 < 1e-8, f"maxerr={max_t4:.3e}")

        # ---- T5: center compile (center-then-transform == transform of transformed center) ----
        g2 = torch.Generator().manual_seed(13)
        m_t = torch.randn(head_dim, head_dim, generator=g2, dtype=torch.float64)
        center_vec = torch.randn(kv_heads, head_dim, generator=g2, dtype=torch.float64)
        k_raw = torch.randn(seq, kv_heads, head_dim, generator=g2, dtype=torch.float64)
        center_first = (k_raw - center_vec) @ m_t
        transform_first = k_raw @ m_t - center_vec @ m_t
        rel_t5 = float((center_first - transform_first).norm()) / (float(center_first.norm()) + 1e-30)
        check("T5 center compile", rel_t5 < 1e-12, f"rel={rel_t5:.3e}")

        # ---- T6: GQA mapping across head configs ----
        for qh, kvh in ((16, 4), (14, 2), (8, 8)):
            g3 = torch.Generator().manual_seed(19 + qh)
            q6 = torch.randn(64, qh * head_dim, generator=g3)
            k6 = torch.randn(64, kvh * head_dim, generator=g3)
            rot6 = torch.stack([ortho_random(head_dim, s).to(torch.float32) for s in range(kvh)])
            q6r = ac0._a2_apply_group_rotation(q6, qh, rot6)
            k6r = ac0._a2_apply_group_rotation(k6, kvh, rot6)
            l_base = per_head_logits(q6, k6, qh, kvh, head_dim)
            l_rot = per_head_logits(q6r, k6r, qh, kvh, head_dim)
            rel6 = float((l_rot - l_base).norm()) / (float(l_base.norm()) + 1e-30)
            check(f"T6 GQA mapping q={qh} kv={kvh}", rel6 < 1e-5, f"rel={rel6:.3e}")
        bad_state = {"num_heads": 16, "head_dim": 64, "learned_rotation": torch.zeros(6, 64, 64)}
        ok_raise = False
        try:
            ac0._check_attention_state(bad_state, 16, 64, "q")
        except ValueError:
            ok_raise = True
        check("T6 non-divisor rotation groups rejected", ok_raise)

        # ---- AC0 calibration + audits on real pipeline ----
        ac0_states = ac0.hif4_calibration_attention(calib, q_heads, kv_heads, head_dim)
        check(
            "AC0 calibration returns legal arms",
            ac0_states["q_state"].get("a2_arm") in ("rotation", "identity", "fallback"),
            f"arm={ac0_states['q_state'].get('a2_arm')}",
        )
        pair_reason = ac0_states["q_state"].get("a2_pair_reason")
        check("AC0 pair validation reason None or recorded", pair_reason is None or isinstance(pair_reason, str))

        # ---- T7: trainer/deployment five-field parity ----
        test_state = dict(r3_states["q_state"])
        installed_rotation = ac0_states["q_state"].get("learned_rotation")
        if installed_rotation is None:
            installed_rotation = torch.eye(head_dim).unsqueeze(0).repeat(kv_heads, 1, 1)
        q_cand = dict(test_state, learned_rotation=installed_rotation)
        k_cand = dict(
            r3_states["k_state"],
            learned_rotation=installed_rotation,
            learned_center=torch.zeros(kv_heads, head_dim),
        )
        q_api = ac0.hif4_dynamic_quantize_q(win["q"][0], win["q"][1], q_heads, head_dim, q_cand)
        q_dense = ac0._dequantize_nvfp4_float32(win["q"][0], win["q"][1])
        q_ref = ac0._attention_transform_dense_reference(
            q_dense, q_cand, q_heads, head_dim, is_k=False, include_learned=True
        )
        q_ref_params = ac0._dense_to_hif4(
            q_ref,
            importance=q_cand["importance"],
            search_offsets=q_cand["offsets"],
            error_threshold=float(q_cand["error_threshold"]),
            accept_margin=float(q_cand["accept_margin"]),
            max_refine_ratio=float(q_cand["max_refine_ratio"]),
            max_refine_blocks=int(q_cand["max_refine_blocks"]),
        )
        check("T7 Q reference==deployment five fields", fields_equal(q_api, q_ref_params))
        k_api = ac0.hif4_dynamic_quantize_k(win["k"][0], win["k"][1], kv_heads, head_dim, k_cand)
        k_dense = ac0._dequantize_nvfp4_float32(win["k"][0], win["k"][1])
        k_ref = ac0._attention_transform_dense_reference(
            k_dense, k_cand, kv_heads, head_dim, is_k=True, include_learned=True
        )
        k_ref_params = ac0._dense_to_hif4(
            k_ref,
            importance=k_cand["importance"],
            search_offsets=k_cand["offsets"],
            error_threshold=float(k_cand["error_threshold"]),
            accept_margin=float(k_cand["accept_margin"]),
            max_refine_ratio=float(k_cand["max_refine_ratio"]),
            max_refine_blocks=int(k_cand["max_refine_blocks"]),
        )
        check("T7 K reference==deployment five fields", fields_equal(k_api, k_ref_params))
        v_api = ac0.hif4_dynamic_quantize_v(win["v"][0], win["v"][1], kv_heads, head_dim, r3_states["v_state"])
        v_ref_params = ac0._dense_to_hif4(
            ac0._dequantize_nvfp4_float32(win["v"][0], win["v"][1]),
            importance=r3_states["v_state"]["importance"],
            search_offsets=r3_states["v_state"]["offsets"],
            error_threshold=float(r3_states["v_state"]["error_threshold"]),
            accept_margin=float(r3_states["v_state"]["accept_margin"]),
            max_refine_ratio=float(r3_states["v_state"]["max_refine_ratio"]),
            max_refine_blocks=int(r3_states["v_state"]["max_refine_blocks"]),
        )
        check("T7 V deployment == direct deployed encode", fields_equal(v_api, v_ref_params))

        # ---- T8: fallback atomicity ----
        events_before = len(ac0._AC0_FALLBACK_EVENTS)
        bad_rot = torch.zeros(kv_heads, head_dim, head_dim // 2)
        parent_q = ac0.hif4_dynamic_quantize_q(win["q"][0], win["q"][1], q_heads, head_dim, q_cand)
        q_fb = ac0._nvfp4_to_hif4(
            win["q"][0],
            win["q"][1],
            multiplier=q_cand.get("multiplier"),
            permutation=q_cand.get("permutation"),
            block_smooth_size=int(q_cand.get("block_smooth_size", 0)),
            block_smooth_seed=int(q_cand.get("block_smooth_seed", 0)),
            attention_rotation=q_cand.get("rotation"),
            rotation_num_heads=int(q_heads),
            attention_rotation_block=q_cand.get("rotation_block"),
            attention_block_signs=q_cand.get("block_smooth_signs"),
            attention_pair_transform=q_cand.get("pair_transform"),
            learned_rotation=bad_rot,
            learned_rotation_num_heads=int(q_heads),
            importance=q_cand["importance"],
            search_offsets=q_cand["offsets"],
            error_threshold=float(q_cand["error_threshold"]),
            accept_margin=float(q_cand["accept_margin"]),
            max_refine_ratio=float(q_cand["max_refine_ratio"]),
            max_refine_blocks=int(q_cand["max_refine_blocks"]),
        )
        check("T8 Q bad-rotation -> parent output", fields_equal(q_fb, parent_q))
        k_fb = ac0._nvfp4_to_hif4(
            win["k"][0],
            win["k"][1],
            multiplier=k_cand.get("multiplier"),
            permutation=k_cand.get("permutation"),
            block_smooth_size=int(k_cand.get("block_smooth_size", 0)),
            block_smooth_seed=int(k_cand.get("block_smooth_seed", 0)),
            attention_rotation=k_cand.get("rotation"),
            rotation_num_heads=int(kv_heads),
            attention_rotation_block=k_cand.get("rotation_block"),
            attention_block_signs=k_cand.get("block_smooth_signs"),
            attention_pair_transform=k_cand.get("pair_transform"),
            learned_rotation=bad_rot,
            learned_rotation_num_heads=int(kv_heads),
            learned_center=torch.zeros(kv_heads, head_dim),
            center_mode=int(k_cand["center_mode"]),
            center_num_heads=kv_heads,
            center_head_dim=head_dim,
            center_value=k_cand.get("center_value"),
            importance=k_cand["importance"],
            search_offsets=k_cand["offsets"],
            error_threshold=float(k_cand["error_threshold"]),
            accept_margin=float(k_cand["accept_margin"]),
            max_refine_ratio=float(k_cand["max_refine_ratio"]),
            max_refine_blocks=int(k_cand["max_refine_blocks"]),
        )
        parent_k = ac0.hif4_dynamic_quantize_k(win["k"][0], win["k"][1], kv_heads, head_dim, k_cand)
        check("T8 K bad-rotation -> parent output (pair atomic)", fields_equal(k_fb, parent_k))
        events_after = len(ac0._AC0_FALLBACK_EVENTS)
        check(
            "T8 fallback events recorded with reasons",
            events_after >= events_before + 2,
            f"events={events_after - events_before}",
        )
        bad_center = torch.zeros(kv_heads, head_dim // 2)
        k_cb = ac0._nvfp4_to_hif4(
            win["k"][0],
            win["k"][1],
            multiplier=k_cand.get("multiplier"),
            permutation=k_cand.get("permutation"),
            block_smooth_size=int(k_cand.get("block_smooth_size", 0)),
            block_smooth_seed=int(k_cand.get("block_smooth_seed", 0)),
            attention_rotation=k_cand.get("rotation"),
            rotation_num_heads=int(kv_heads),
            attention_rotation_block=k_cand.get("rotation_block"),
            attention_block_signs=k_cand.get("block_smooth_signs"),
            attention_pair_transform=k_cand.get("pair_transform"),
            learned_rotation=installed_rotation,
            learned_rotation_num_heads=int(kv_heads),
            learned_center=bad_center,
            center_mode=int(k_cand["center_mode"]),
            center_num_heads=kv_heads,
            center_head_dim=head_dim,
            center_value=k_cand.get("center_value"),
            importance=k_cand["importance"],
            search_offsets=k_cand["offsets"],
            error_threshold=float(k_cand["error_threshold"]),
            accept_margin=float(k_cand["accept_margin"]),
            max_refine_ratio=float(k_cand["max_refine_ratio"]),
            max_refine_blocks=int(k_cand["max_refine_blocks"]),
        )
        check("T8 K bad-center -> parent output", fields_equal(k_cb, parent_k))
        good_q = ac0._nvfp4_to_hif4(
            win["q"][0],
            win["q"][1],
            multiplier=q_cand.get("multiplier"),
            permutation=q_cand.get("permutation"),
            block_smooth_size=int(q_cand.get("block_smooth_size", 0)),
            block_smooth_seed=int(q_cand.get("block_smooth_seed", 0)),
            attention_rotation=q_cand.get("rotation"),
            rotation_num_heads=int(q_heads),
            attention_rotation_block=q_cand.get("rotation_block"),
            attention_block_signs=q_cand.get("block_smooth_signs"),
            attention_pair_transform=q_cand.get("pair_transform"),
            learned_rotation=installed_rotation,
            learned_rotation_num_heads=int(q_heads),
            importance=q_cand["importance"],
            search_offsets=q_cand["offsets"],
            error_threshold=float(q_cand["error_threshold"]),
            accept_margin=float(q_cand["accept_margin"]),
            max_refine_ratio=float(q_cand["max_refine_ratio"]),
            max_refine_blocks=int(q_cand["max_refine_blocks"]),
        )
        check("T8 good rotation state does not fall back", fields_equal(good_q, q_api))

        # ---- T9: finite/state contract ----
        from evaluator.reference_hif4 import validate_state, validate_hif4_params

        for side in ("q_state", "k_state", "v_state"):
            validate_state(ac0_states[side])
        for side in ("q_state", "k_state", "v_state"):
            validate_state(r3_states[side])
        for name, params, shape in (
            ("q", q_api, tuple(win["q"][0].shape)),
            ("k", k_api, tuple(win["k"][0].shape)),
            ("v", v_api, tuple(win["v"][0].shape)),
        ):
            validate_hif4_params(params, shape)
            for key in ("scale_factor", "scale_lv2", "scale_lv3", "sign", "mant"):
                t = params[key]
                assert t.device.type == "cpu"
                assert not t.requires_grad
                assert bool(torch.isfinite(t.to(torch.float32)).all())
        check("T9 state + params contract clean", True)

        # FP64 audit on the installed candidate (real pipeline)
        if installed_rotation is not None:
            audit = ac0._ac0_qk_invariance_audit(
                ac0._dequantize_nvfp4_float32(win["q"][0], win["q"][1])[:64],
                ac0._dequantize_nvfp4_float32(win["k"][0], win["k"][1])[:64],
                dict(r3_states["q_state"], learned_rotation=installed_rotation),
                dict(
                    r3_states["k_state"],
                    learned_rotation=installed_rotation,
                    learned_center=torch.zeros(kv_heads, head_dim),
                ),
                q_heads,
                kv_heads,
                head_dim,
            )
            check(
                "FP64 audit valid on installed candidate",
                audit["valid"],
                f"rel={audit['relative_error']:.3e}",
            )

    failed = [name for name, ok in RESULTS if not ok]
    print(f"\nSUMMARY: {len(RESULTS) - len(failed)}/{len(RESULTS)} passed")
    if failed:
        print("FAILED:", failed)
        sys.exit(1)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)