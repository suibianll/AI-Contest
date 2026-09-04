"""Build a diagnostic copy of the v188 candidate that dumps the Jacobian
sensitivity LOO gate internals per attention calibration call.

Dump file: diag_v188_gate.jsonl (repo root, one JSON line per layer call).
Pure instrumentation: no numerical change to the candidate.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
src = (ROOT / "solution.py").read_text(encoding="utf-8")

# 1) init the per-layer gate record alongside the delta list
anchor_b = "        sensitivity_deltas = []"
assert src.count(anchor_b) == 1, "anchor B not unique"
src = src.replace(
    anchor_b,
    anchor_b
    + """
        _diag_gate = {
            "parent_causal": list(parent_causal),
            "parent_safety": list(parent_safety),
            "folds": [],
        }""",
    1,
)

# 2) collect held-out candidate losses per fold
anchor_a = """            sensitivity_deltas.append(
                (parent_causal[held_out] - held_causal[0])
                / max(parent_causal[held_out], _EPS)
            )
            sensitivity_deltas.append(
                (parent_safety[held_out] - held_safety[0])
                / max(parent_safety[held_out], _EPS)
            )"""
assert src.count(anchor_a) == 1, "anchor A not unique"
src = src.replace(
    anchor_a,
    anchor_a
    + """
            _diag_gate["folds"].append(
                {
                    "held_causal": held_causal[0],
                    "held_safety": held_safety[0],
                }
            )""",
    1,
)

# 3) dump the record after the accept decision, before return
anchor_c = """                k_state["importance"] = _cpu_state_tensor(
                    _aggregate_sensitivity(k_folds).reshape(-1)
                )
    return {"q_state": q_state, "k_state": k_state, "v_state": v_state}"""
assert src.count(anchor_c) == 1, "anchor C not unique"
repl_c = """                k_state["importance"] = _cpu_state_tensor(
                    _aggregate_sensitivity(k_folds).reshape(-1)
                )
        _diag_accepted = False
        if len(sensitivity_deltas) >= 2:
            _diag_accepted = bool(
                float(delta_tensor.median()) > _ATTENTION_SENSITIVITY_MIN_GAIN
                and float(delta_tensor.min())
                > -_ATTENTION_SENSITIVITY_WORST_TOLERANCE
            )
        _diag_gate["accepted"] = _diag_accepted
        _diag_gate["deltas"] = sensitivity_deltas
        _diag_q_imp = _aggregate_sensitivity(q_folds).repeat_interleave(
            group_size, dim=0
        ).reshape(-1)
        _diag_k_imp = _aggregate_sensitivity(k_folds).reshape(-1)
        _diag_q_parent = q_state["importance"].to(
            device=_diag_q_imp.device, dtype=torch.float32
        ).reshape(-1)
        _diag_k_parent = k_state["importance"].to(
            device=_diag_k_imp.device, dtype=torch.float32
        ).reshape(-1)
        _diag_gate["q_rel_dist"] = float(
            (_diag_q_imp - _diag_q_parent).norm().item()
            / max(float(_diag_q_parent.norm().item()), _EPS)
        )
        _diag_gate["k_rel_dist"] = float(
            (_diag_k_imp - _diag_k_parent).norm().item()
            / max(float(_diag_k_parent.norm().item()), _EPS)
        )
        _diag_gate["q_imp_range"] = [
            float(_diag_q_imp.min().item()),
            float(_diag_q_imp.max().item()),
        ]
        _diag_gate["k_imp_range"] = [
            float(_diag_k_imp.min().item()),
            float(_diag_k_imp.max().item()),
        ]
        _diag_gate["q_parent_range"] = [
            float(_diag_q_parent.min().item()),
            float(_diag_q_parent.max().item()),
        ]
        _diag_gate["k_parent_range"] = [
            float(_diag_k_parent.min().item()),
            float(_diag_k_parent.max().item()),
        ]
        import json as _diag_json

        with open("diag_v188_gate.jsonl", "a", encoding="utf-8") as _diag_f:
            _diag_f.write(_diag_json.dumps(_diag_gate) + "\\n")
    return {"q_state": q_state, "k_state": k_state, "v_state": v_state}"""
src = src.replace(anchor_c, repl_c, 1)

out = ROOT / "logs" / "execution" / "diag_v188_gate.py"
out.write_text(src, encoding="utf-8")
print("written", out)
