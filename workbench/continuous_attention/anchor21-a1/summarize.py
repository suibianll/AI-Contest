"""Pair exact cases and apply this workpackage's gate, not the old total-L1 gate."""
from pathlib import Path
import hashlib
import json
import statistics as st

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
OUT = ROOT / "artifacts/proxy_v3/continuous/attention/anchor21-a1"


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def stats(rows):
    deltas = [r["delta"] for r in rows]
    ordered = sorted(deltas)
    quartiles = st.quantiles(deltas, n=4, method="inclusive")
    return {"cases": len(rows), "parent_mean": st.mean(r["parent_gain"] for r in rows),
            "candidate_mean": st.mean(r["candidate_gain"] for r in rows), "delta_mean": st.mean(deltas),
            "delta_median": st.median(deltas), "q25": quartiles[0], "q75": quartiles[2],
            "worst_quartile_delta_mean": st.mean(ordered[:max(1, len(rows)//4)]),
            "negative_L1": st.mean(max(-v, 0) for v in deltas), "total_L1": st.mean(abs(v) for v in deltas),
            "positive": sum(v > 0 for v in deltas), "negative": sum(v < 0 for v in deltas), "zero": sum(v == 0 for v in deltas)}


def panel(name):
    rows = []
    for shard in range(6):
        suffix = f"{'ood-' if name == 'ood' else ''}attention-shard{shard}.json"
        parent = load(OUT / name / ("baseline-" + suffix))["results"][0]
        candidate = load(OUT / name / ("candidate-" + suffix))["results"][0]
        assert parent["status"] == candidate["status"] == "ok"
        assert candidate["source_sha256"] == hashlib.sha256((HERE / "solution.py").read_bytes()).hexdigest()
        def key(c):
            return (c["layer"], c["test_window"], c["test_split"], c["test_length"])
        p = {key(c): c for c in parent["case_scores"]["attention"]}
        c = {key(c): c for c in candidate["case_scores"]["attention"]}
        assert len(p) == len(c) == 8 and p.keys() == c.keys()
        for k in p:
            assert p[k]["mse_standard"] == c[k]["mse_standard"]
            assert p[k]["reference_energy"] == c[k]["reference_energy"]
            rows.append({"layer": k[0], "window": k[1], "split": k[2], "length": k[3],
                         "delta": c[k]["gain"] - p[k]["gain"], "parent_gain": p[k]["gain"], "candidate_gain": c[k]["gain"]})
    result = stats(rows)
    result["by_split"] = {k: stats([r for r in rows if r["split"] == k]) for k in sorted({r["split"] for r in rows})}
    result["by_length"] = {str(k): stats([r for r in rows if r["length"] == k]) for k in sorted({r["length"] for r in rows})}
    by_layer = [(k, st.mean(r["delta"] for r in rows if r["layer"] == k)) for k in sorted({r["layer"] for r in rows})]
    result["worst_layers"] = sorted(by_layer, key=lambda x: x[1])[:5]
    signs = []
    for layer in range(24):
        pair = [r["delta"] for r in rows if r["layer"] == layer]
        assert len(pair) == 2
        signs.append((pair[0] > 0) == (pair[1] > 0) and (pair[0] < 0) == (pair[1] < 0))
    result["validation_test_sign_agreement_including_ties"] = st.mean(signs)
    result["rows"] = rows
    return result


identity, ood = panel("id"), panel("ood")
timing = load(OUT / "timing/default.json")["results"][0]
seconds = timing["timing"]["api_seconds"]
predicted = 170.3 + .115*seconds["hif4_calibration_and_quantize_weight"] + .694*seconds["hif4_calibration_attention"] + .734*seconds["hif4_dynamic_quantize_activation"] - 1.58*sum(seconds["hif4_dynamic_quantize_"+r] for r in "qkv")
diagnostic = load(HERE / "diagnostics.json")
eligible = identity["negative_L1"] < 0.02 and predicted < 280 and diagnostic["accepted_layers"] > 0
assert timing["score"]["linear_cases"] == 168 and timing["score"]["attention_cases"] == 120
assert timing["score"]["linear_mean"] == 0
split_positive = all(s["delta_mean"] > 0 for s in identity["by_split"].values())
result = {"run_id": "anchor21-a1", "source_sha256": hashlib.sha256((HERE / "solution.py").read_bytes()).hexdigest(),
          "status": "READY_FOR_OFFICIAL_EXPLORATION" if eligible else "REJECTED_LOCAL_GATE",
          "local_class": "POSITIVE" if identity["delta_mean"] > 0 and split_positive else "EXPLORATORY_SPLIT_OR_MEAN_NEGATIVE",
          "official_score": None, "official_time_s": None, "official_status": "unregistered/NA",
          "id": identity, "ood": ood, "ood_delta_gap": identity["delta_mean"] - ood["delta_mean"],
          "ood_is_veto": False, "fresh_default": {"score": timing["score"], "timing": timing["timing"], "predicted_official_s": predicted},
          "attempted_layers": diagnostic["attempted_layers"], "accepted_layers": diagnostic["accepted_layers"],
          "mean_q_scale_ratio2": diagnostic["mean_q_scale_ratio2"], "mean_k_scale_ratio2": diagnostic["mean_k_scale_ratio2"],
          "cross_model": load(OUT / "cross/gpt2.json")["result"]["score"],
          "controls": {"linear": "exact R3 source prefix; 168 fresh-default cases zero error gain vs standard", "V": diagnostic["V_control"]},
          "evaluator_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in [ROOT / "evaluator/eval.py", ROOT / "evaluator/eval_system.py", ROOT / "evaluator/proxy_v3_eval.py", ROOT / "evaluator/official_eval.py", ROOT / "evaluator/reference_hif4.py"]},
          "next": "Await distinct-SHA official result; prepare A21-3 scale/output mismatch card, no parameter sweep."}
(HERE / "manifest.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
for name in ["id", "ood"]:
    print(name, json.dumps({k: v for k, v in result[name].items() if k not in ["rows", "by_split", "by_length"]}))
    print('splits', {k:v['delta_mean'] for k,v in result[name]['by_split'].items()})
print('time_prediction', predicted, 'delta_gap', result['ood_delta_gap'], 'status', result['status'], result['local_class'])
