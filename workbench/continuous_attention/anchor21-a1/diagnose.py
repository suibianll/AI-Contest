"""Read cached calibration states and diagnose deployed outputs; no retraining."""
from pathlib import Path
import hashlib
import importlib.util
import json
import math
import statistics
import sys
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
OUT = ROOT / "artifacts/proxy_v3/continuous/attention/anchor21-a1"
sys.path.insert(0, str(ROOT / "evaluator"))
sys.path.insert(0, str(ROOT / "workbench/v162_attention"))
import official_eval as ev
import proxy_v3_eval as v3
import gpu_lock


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    assert gpu_lock.acquire("A", "anchor21-a1-diagnostics") == 0
    try:
        candidate = module(HERE / "solution.py", "a21_diagnostic")
        parent = module(ROOT / "workbench/v162_attention/candidate_v3d/solution.py", "r3_diagnostic")
        raw = ev.load_pack(ROOT / "artifacts/official_eval/cache/qwen2.5-0.5b-proxy-v2.pt")
        device = torch.device("cuda")
        records, audits = [], []
        with torch.inference_mode():
            for shard in range(6):
                pack = v3.prepare_shard(raw, shard, "attention", False)
                docs = [load(OUT / "id" / f"{kind}-attention-shard{shard}.json")["results"][0] for kind in ["baseline", "candidate"]]
                states = []
                for doc in docs:
                    _, st = v3.load_calibration_artifact(Path(doc["timing"]["calibration_artifact"]), doc["timing"]["calibration_identity"], pack)
                    states.append(st)
                for layer, st in states[1].items():
                    audits.append({"layer": layer, **{k: val for k, val in st["q_state"].items() if k.startswith("a21_")}})
                for case in pack.attention_cases:
                    pairs = [ev._move_pair(pair, device) for pair in pack.test_qkv[case.test_window][case.layer]]
                    refs = [candidate._dequantize_nvfp4_float32(*p).float()[None] for p in pairs]
                    target, target_logits, target_prob = ev._attention_trace(*refs, pack.q_heads, pack.kv_heads, pack.head_dim)
                    row = {"layer": case.layer, "window": case.test_window, "split": pack.test_windows[case.test_window].split, "length": refs[0].shape[1]}
                    all_params = []
                    for name, mod, st in zip(["parent", "candidate"], [parent, candidate], states):
                        outputs, params_list = [], []
                        measurements = {}
                        for role, heads, pair in zip(["q", "k", "v"], [pack.q_heads, pack.kv_heads, pack.kv_heads], pairs):
                            captured = []
                            original = mod._dense_to_hif4
                            def capture(dense, *args, **kwargs):
                                captured.append(dense.clone())
                                return original(dense, *args, **kwargs)
                            mod._dense_to_hif4 = capture
                            try:
                                params = getattr(mod, "hif4_dynamic_quantize_" + role)(*pair, heads, pack.head_dim, st[case.layer][role + "_state"])
                            finally:
                                mod._dense_to_hif4 = original
                            decoded = mod._dequantize_hif4(params).float()
                            outputs.append(decoded[None])
                            params_list.append(params)
                            if role != "v":
                                pre = captured[0]
                                amax = pre.reshape(-1, 64).abs().amax(-1)
                                cap = (params["scale_factor"] * params["scale_lv2"] * params["scale_lv3"] * 1.75).expand_as(params["mant"]).reshape_as(pre)
                                measurements[role] = {
                                    "pre_amax_mean": float(amax.mean()), "pre_amax_q90": float(torch.quantile(amax, 0.9)),
                                    "e6m2_mean": float(params["scale_factor"].float().mean()),
                                    "e6m2_q90": float(torch.quantile(params["scale_factor"].float().flatten(), 0.9)),
                                    "beyond_selected_hierarchy_capacity_fraction": float((pre.abs() > cap).float().mean()),
                                    "same_coordinate_quant_mse": float((pre - decoded).square().mean()),
                                }
                        output, logits, prob = ev._attention_trace(*outputs, pack.q_heads, pack.kv_heads, pack.head_dim)
                        dl = logits - target_logits
                        # K centering gives a harmless row-constant shift. Report
                        # both raw and row-centered logits, never conflate them.
                        measurements.update(logit_mse=float(dl.square().mean()),
                            centered_logit_mse=float((dl - dl.mean(-1, keepdim=True)).square().mean()),
                            probability_mse=float((prob - target_prob).square().mean()),
                            output_mse=float((output - target).square().mean()))
                        row[name] = measurements
                        all_params.append(params_list)
                    row["changed_fraction"] = {role: {key: float((a[key] != b[key]).float().mean()) for key in a}
                        for role, a, b in zip(["q", "k", "v"], *all_params)}
                    assert all(value == 0 for value in row["changed_fraction"]["v"].values())
                    for name, doc in zip(["parent", "candidate"], docs):
                        recorded = next(c for c in doc["case_scores"]["attention"] if c["layer"] == case.layer and c["test_window"] == case.test_window)
                        assert math.isclose(row[name]["output_mse"], recorded["mse_player"], rel_tol=1e-5, abs_tol=1e-10)
                    records.append(row)
        result = {"source_sha256": hashlib.sha256((HERE / "solution.py").read_bytes()).hexdigest(),
                  "protocol": "diagnostic-cached-state-replay-not-default-timing", "cases": records, "audit": audits,
                  "accepted_layers": sum(a["a21_accepted_groups"] > 0 for a in audits),
                  "attempted_layers": len(audits), "V_control": "bitwise identical all 48 cases",
                  "mean_q_scale_ratio2": statistics.mean(a["a21_q_scale_ratio2"] for a in audits),
                  "mean_k_scale_ratio2": statistics.mean(a["a21_k_scale_ratio2"] for a in audits)}
        (HERE / "diagnostics.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({k: v for k, v in result.items() if k not in ["cases", "audit"]}, indent=2))
    finally:
        gpu_lock.release("A", "anchor21-a1-diagnostics")


if __name__ == "__main__":
    main()
