"""C-FR1 verification on real calibration windows.

Every claim the card makes is checked here rather than asserted:

  A. the candidate is standalone and exposes the six APIs;
  B. the reciprocal transform is exact -- the *continuous* QK product is
     unchanged at the candidate's compiled state, in float64;
  C. V is untouched, and V's params are bitwise identical to the parent's;
  D. the compiled state is bitwise the parent's whenever the gate rejects, so
     a rejected candidate cannot differ from the root in any way;
  E. reachability: how many Q/K mantissa codes actually move, per layer;
  F. what the gate saw, per layer.

CPU only.  No shard, no candidate submission.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
ARCHIVE = ROOT / "solutions/20260910_v237_linear-tf2-first-pass-gradient-reuse_scoreNA_timeNA/solution.py"
CANDIDATE = HERE / "candidate" / "solution.py"
LAYERS = (0, 1, 5, 8, 15, 22)


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def run() -> int:
    torch.set_num_threads(4)
    torch.set_grad_enabled(False)
    parent = load("cfr_parent", ARCHIVE)
    candidate = load("cfr_candidate", CANDIDATE)

    sys.path.insert(0, str(ROOT / "evaluator"))
    import official_eval as v2  # noqa: PLC0415

    pack = torch.load(
        ROOT / "artifacts/official_eval/cache/qwen3.5-4b-proxy-v2.pt",
        map_location="cpu", mmap=True, weights_only=False,
    )
    q_heads = int(pack["q_heads"])
    kv_heads = int(pack["kv_heads"])
    head_dim = int(pack["head_dim"])
    device = torch.device("cpu")

    apis = (
        "hif4_calibration_and_quantize_weight",
        "hif4_dynamic_quantize_activation",
        "hif4_calibration_attention",
        "hif4_dynamic_quantize_q",
        "hif4_dynamic_quantize_k",
        "hif4_dynamic_quantize_v",
    )
    missing = [name for name in apis if not hasattr(candidate, name)]
    print(f"[A] six APIs present: {not missing}  missing={missing}")
    if missing:
        return 1

    def dense_logits(q_params, k_params, q_ref_shape, k_ref_shape):
        q = v2.dequantize_hif4(v2._cpu_params(q_params), q_ref_shape).double()
        k = v2.dequantize_hif4(v2._cpu_params(k_params), k_ref_shape).double()
        qh = q.reshape(-1, q_heads, head_dim).transpose(0, 1)
        kh = k.reshape(-1, kv_heads, head_dim).transpose(0, 1)
        kh = kh.repeat_interleave(q_heads // kv_heads, dim=0)
        return qh @ kh.transpose(-1, -2)

    rows = []
    for layer in LAYERS:
        windows = [
            {
                role: tuple(
                    t.to(device)
                    for t in v2._pair(pack["calibration_qkv"][s][layer][i].to(torch.float32))
                )
                for i, role in enumerate(("q", "k", "v"))
            }
            for s in range(len(pack["calibration_qkv"]))
        ]
        parent_states = parent.hif4_calibration_attention(windows, q_heads, kv_heads, head_dim)
        cand_states = candidate.hif4_calibration_attention(windows, q_heads, kv_heads, head_dim)

        # G. the design invariant itself, independent of any downstream effect:
        #    M^T A M and M^{-1} B M^{-T} must both equal diag(lam)^{1/2}, i.e.
        #    equal each other, up to the ridge.  This checks the algebra rather
        #    than the consequence, so an implementation slip cannot hide behind
        #    a gate that happens to reject.
        invariant_gap = None
        try:
            fit = windows[: len(windows) - 2]
            aq, ak = candidate._cfr_moments(
                fit, parent_states, q_heads, kv_heads, head_dim, device
            )
            m, m_inv_t = candidate._cfr_matrix(aq, ak, candidate._CFR_RIDGE)
            m_inv = m_inv_t.transpose(-1, -2)
            dim = int(head_dim)
            eye = torch.eye(dim, dtype=m.dtype)
            rq_d = aq.diagonal(dim1=-2, dim2=-1).mean(-1, keepdim=True).clamp_min(1e-12).unsqueeze(-1)
            rk_d = ak.diagonal(dim1=-2, dim2=-1).mean(-1, keepdim=True).clamp_min(1e-12).unsqueeze(-1)
            aq_r = aq + (candidate._CFR_RIDGE * rq_d) * eye
            ak_r = ak + (candidate._CFR_RIDGE * rk_d) * eye
            left = m.transpose(-1, -2) @ aq_r @ m
            right = m_inv @ ak_r @ m_inv.transpose(-1, -2)
            invariant_gap = float((left - right).abs().max() / max(float(left.abs().max()), 1e-30))
            # the closed-form inverse must actually be the inverse
            identity_gap = float(
                (m_inv @ m - torch.eye(dim, dtype=m.dtype)).abs().max()
            )
        except Exception as exc:  # noqa: BLE001 - report, never mask
            invariant_gap = f"error: {type(exc).__name__}: {exc}"
            identity_gap = None

        # B/C/E must be measured on the *proposed* transform, not on the state
        # the calibration returned.  When the gate rejects, the returned state is
        # the parent's bit for bit, so measuring there would report "no codes
        # moved" for every rejected layer -- a vacuous reading that hides an
        # implementation slip behind a rejection.
        proposed = None
        try:
            fit = windows[: len(windows) - 2]
            aq, ak = candidate._cfr_moments(
                fit, parent_states, q_heads, kv_heads, head_dim, device
            )
            m, m_inv_t = candidate._cfr_matrix(aq, ak, candidate._CFR_RIDGE)
            rot_q, rot_k, center_k = candidate._cfr_compile(
                parent_states, m, m_inv_t, kv_heads, head_dim
            )
            proposed = candidate._cfr_parent_copy(parent_states)
            proposed["q_state"]["learned_rotation"] = rot_q
            proposed["k_state"]["learned_rotation"] = rot_k
            if center_k is not None:
                proposed["k_state"]["learned_center"] = center_k
        except Exception as exc:  # noqa: BLE001 - report, never mask
            print(f"[L{layer:>2}] proposed-state construction failed: {exc}", flush=True)

        arm = cand_states["q_state"].get("cfr_arm")
        accepted = int(cand_states["q_state"].get("cfr_accepted", 0))

        # D. rejected -> the compiled state must be the parent's, bitwise.
        same_rotation = True
        same_center = True
        for role in ("q_state", "k_state"):
            for key in ("learned_rotation", "learned_center"):
                pv = parent_states[role].get(key)
                cv = cand_states[role].get(key)
                if (pv is None) != (cv is None):
                    same_rotation = False
                    continue
                if pv is not None and not torch.equal(pv, cv):
                    if key == "learned_center":
                        same_center = False
                    else:
                        same_rotation = False
        state_matches_parent = same_rotation and same_center

        # B/C/E on the first test window that exists for this layer.
        entry = pack["test_qkv"][0][layer]
        logit_gap = None
        invariance_gap = None
        q_moved = k_moved = None
        v_bitwise = None
        if entry is not None and proposed is not None:
            pairs = [v2._pair(entry[i].to(torch.float32)) for i in range(3)]
            q_ref_shape, k_ref_shape, v_ref_shape = (
                v2.dequantize_nvfp4(*p).shape for p in pairs
            )
            p_q = parent.hif4_dynamic_quantize_q(*pairs[0], q_heads, head_dim, parent_states["q_state"])
            p_k = parent.hif4_dynamic_quantize_k(*pairs[1], kv_heads, head_dim, parent_states["k_state"])
            p_v = parent.hif4_dynamic_quantize_v(*pairs[2], kv_heads, head_dim, parent_states["v_state"])
            c_q = candidate.hif4_dynamic_quantize_q(*pairs[0], q_heads, head_dim, proposed["q_state"])
            c_k = candidate.hif4_dynamic_quantize_k(*pairs[1], kv_heads, head_dim, proposed["k_state"])
            c_v = candidate.hif4_dynamic_quantize_v(*pairs[2], kv_heads, head_dim, proposed["v_state"])
            # Reachability in logit space: the *quantized* operands' product.
            logit_gap = float(
                (dense_logits(p_q, p_k, q_ref_shape, k_ref_shape)
                 - dense_logits(c_q, c_k, q_ref_shape, k_ref_shape)).abs().max()
            )
            # B. The reciprocal invariance, measured on the PRE-quantization
            #    dense operands.  Measuring it on the decoded codes -- as the
            #    line above does -- would report a large gap even for an exact
            #    transform, because the codes are exactly what is supposed to
            #    change.  These are two different readings and both are kept.
            refs_dense = [v2.dequantize_nvfp4(*q).to(torch.float32) for q in pairs]

            def _logits_from_state(state):
                qc = candidate._cfr_parent_coordinate(
                    refs_dense[0], state["q_state"], q_heads, head_dim, False
                ).double()
                kc = candidate._cfr_parent_coordinate(
                    refs_dense[1], state["k_state"], kv_heads, head_dim, True
                ).double()
                qh = qc.reshape(-1, q_heads, head_dim).transpose(0, 1)
                kh = kc.reshape(-1, kv_heads, head_dim).transpose(0, 1)
                kh = kh.repeat_interleave(q_heads // kv_heads, dim=0)
                return qh @ kh.transpose(-1, -2)

            invariance_gap = float(
                (_logits_from_state(parent_states) - _logits_from_state(proposed)).abs().max()
            )
            q_moved = int((c_q["mant"] != p_q["mant"]).sum())
            k_moved = int((c_k["mant"] != p_k["mant"]).sum())
            v_bitwise = all(torch.equal(c_v[key], p_v[key]) for key in p_v)

        rows.append(
            {
                "layer": layer,
                "cfr_arm": arm,
                "cfr_accepted": accepted,
                "gate_parent": cand_states["q_state"].get("cfr_gate_parent"),
                "gate_candidate": cand_states["q_state"].get("cfr_gate_candidate"),
                "singular_min": cand_states["q_state"].get("cfr_singular_min"),
                "singular_max": cand_states["q_state"].get("cfr_singular_max"),
                "identity_distance": cand_states["q_state"].get("cfr_identity_distance"),
                "state_matches_parent": state_matches_parent,
                "invariant_gap": invariant_gap,
                "inverse_gap": identity_gap,
                "logit_gap_quantized": logit_gap,
                "invariance_gap_fp64": invariance_gap,
                "q_mant_moved": q_moved,
                "k_mant_moved": k_moved,
                "v_bitwise_same": v_bitwise,
                "error": cand_states["q_state"].get("cfr_error"),
            }
        )
        print(
            f"[L{layer:>2}] arm={arm} accepted={accepted} "
            f"state==parent:{state_matches_parent} "
            f"|dlogit|q={logit_gap if logit_gap is None else f'{logit_gap:.3e}'} "
            f"|dlogit|dense={invariance_gap if invariance_gap is None else f'{invariance_gap:.3e}'} "
            f"q_moved={q_moved} k_moved={k_moved} v_bitwise={v_bitwise} "
            f"invariant={invariant_gap if isinstance(invariant_gap, str) else f'{invariant_gap:.3e}'} "
            f"gate={cand_states['q_state'].get('cfr_gate_parent')} -> "
            f"{cand_states['q_state'].get('cfr_gate_candidate')}"
            + (f"  error={rows[-1]['error']}" if rows[-1]["error"] else ""),
            flush=True,
        )

    dense_gaps = [r["invariance_gap_fp64"] for r in rows if r["invariance_gap_fp64"] is not None]
    quant_gaps = [r["logit_gap_quantized"] for r in rows if r["logit_gap_quantized"] is not None]
    print()
    print(
        f"[B] reciprocity (pre-quantization, fp64): max |dlogit| = "
        f"{max(dense_gaps) if dense_gaps else float('nan'):.3e}"
    )
    print(
        f"[B'] reachability in logit space (quantized): max |dlogit| = "
        f"{max(quant_gaps) if quant_gaps else float('nan'):.3e}"
    )
    print(
        f"[C] V bitwise identical on every layer: "
        f"{all(r['v_bitwise_same'] for r in rows if r['v_bitwise_same'] is not None)}"
    )
    rejected = [r for r in rows if r["cfr_accepted"] == 0]
    print(
        f"[D] rejected layers whose state equals the parent bitwise: "
        f"{sum(1 for r in rejected if r['state_matches_parent'])}/{len(rejected)}"
    )
    moved = [r for r in rows if (r["q_mant_moved"] or 0) + (r["k_mant_moved"] or 0) > 0]
    print(f"[E] reachability: layers where Q/K mantissa moved = {len(moved)}/{len(rows)}")
    print(f"[F] accepted = {sum(r['cfr_accepted'] for r in rows)}/{len(rows)}")

    (HERE / "verify.json").write_text(
        json.dumps({"layers": list(LAYERS), "rows": rows}, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {HERE / 'verify.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
