"""Attention-side error ledger (continuous-research-loop R-1).

Grid (2026-09-08-continuous-research-loop.md section 2.3):
  F1 Q/K-side total error : deploy (transform+quantize) Q/K attention output
                            vs NVFP4 float reference output, V fixed at the
                            parent's quantized V; normalized by mse_standard
                            of the paired 4B case.
  F2 quantization-code error : same T-coordinate comparison:
                            attn(Q_hat, K_hat, v_hat) vs attn(T(Q_ref),
                            T(K_ref), v_hat) -- pure code/scale allocation
                            cost, transform roundtrip exact by construction.
  F3a logits error        : ||Q_hat K_hat^T - T(Q)T(K)^T||^2 / sqrt(d)^2
                            (T coordinates).
  F3b probability error   : ||softmax(Q_hat K_hat^T) - softmax(T(Q)T(K)^T)||^2.
  F4 final output error   : mean(mse_player / mse_standard) over the 72 paired
                            4B cases (zero API, from archived JSON).
  F5 nonlinear/distribution residual : F4 - F1 (softmax coupling, length and
                            layer distribution, V quantization).
  F1 - F2                 : residual of the transform choice itself
                            (center/GQA constraint, inverse alignment).

Coverage: 6 FA layers x 12 test windows = 72 cases, exactly paired with the
4B panel JSON cases (layer, test_window). No model forward: everything is
recomputed from the dense NVFP4 cache. Parent = A23 (sha 8714ac2a...92dbf).
"""
from pathlib import Path
import argparse
import importlib.util
import json
import math
import sys

import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "evaluator"))
sys.path.insert(0, str(ROOT / "workbench/v162_attention"))
import official_eval as ev  # noqa: E402
import gpu_lock  # noqa: E402

A23_DIR = ROOT / "artifacts/proxy_v3/continuous/attention/anchor23-a1/id"
A25_DIR = ROOT / "artifacts/proxy_v3/continuous/attention/anchor25-a1/vs_b/vs_b"
PARENT_SOLUTION = ROOT / "workbench/continuous_attention/anchor23-a1/solution.py"
CACHE = ROOT / "artifacts/official_eval/cache/qwen3.5-4b-proxy-v2.pt"
OUT_DIR = ROOT / "artifacts/continuous/attention"
FA_LAYERS = [0, 1, 5, 8, 15, 22]
PARENT_SHA = "8714ac2a044779465c5e406ef0768be7071ac626f3a2171cdc5083350be92dbf"
A23_LOCAL_SHA = "8714ac2a044779465c5e406ef0768be7071ac626f3a2171cdc5083350be92dbf"


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def read_4b_cases(base):
    cases = []
    for shard in range(6):
        d = json.loads((Path(base) / f"candidate-attention-shard{shard}.json").read_text(encoding="utf-8"))
        sha = d["results"][0]["source_sha256"]
        for c in d["results"][0]["case_scores"]["attention"]:
            c["_sha"] = sha
            cases.append(c)
    return cases


def attention_out(mod, q, k, v, qh, kh, dim):
    return mod._a2_attention_forward(q[None].float(), k[None].float(), v[None].float(), qh, kh, dim)[0]


def measure_parent(mod, pack):
    """Per-case F1/F2/F3a/F3b of the A23 parent on all 72 (layer, window)."""
    dev = torch.device("cuda")
    rows = []
    for layer in FA_LAYERS:
        calib = [
            {role: tuple(t.to(dev) for t in ev._pair(dense))
             for role, dense in zip(("q", "k", "v"), pack["calibration_qkv"][f][layer])}
            for f in range(5)
        ]
        with torch.inference_mode():
            states = mod.hif4_calibration_attention(calib, 16, 4, 256)
        for w in range(12):
            item = {role: tuple(t.to(dev) for t in ev._pair(dense))
                    for role, dense in zip(("q", "k", "v"), pack["test_qkv"][w][layer])}
            with torch.inference_mode():
                q_hat = mod._dequantize_hif4(mod.hif4_dynamic_quantize_q(*item["q"], 16, 256, states["q_state"])).float()
                k_hat = mod._dequantize_hif4(mod.hif4_dynamic_quantize_k(*item["k"], 4, 256, states["k_state"])).float()
                v_hat = mod._dequantize_hif4(mod.hif4_dynamic_quantize_v(*item["v"], 4, 256, states["v_state"])).float()
                q_ref = mod._dequantize_nvfp4_float32(*item["q"]).float()
                k_ref = mod._dequantize_nvfp4_float32(*item["k"]).float()
                # T-coordinate float reference (same parent stack, no quantization)
                q_t = mod._a1_stack_transform(q_ref, 16, 256, states["q_state"], False)
                k_t = mod._a1_stack_transform(k_ref, 4, 256, states["k_state"], True)
            out_ref = attention_out(mod, q_ref, k_ref, v_hat, 16, 4, 256)     # F1 numerator ref
            out_player = attention_out(mod, q_hat, k_hat, v_hat, 16, 4, 256)  # deploy
            out_t = attention_out(mod, q_t, k_t, v_hat, 16, 4, 256)           # F2 ref (T coords)
            f1 = float((out_player - out_ref).square().mean())
            f2 = float((out_player - out_t).square().mean())
            dim = 256
            logit_player = (q_hat.reshape(-1, 16, dim) @ k_hat.reshape(-1, 4, dim).transpose(-1, -2)) / math.sqrt(dim)
            logit_t = (q_t.reshape(-1, 16, dim) @ k_t.reshape(-1, 4, dim).transpose(-1, -2)) / math.sqrt(dim)
            f3a = float((logit_player - logit_t).square().mean())
            prob_player = torch.softmax(logit_player, dim=-1)
            prob_t = torch.softmax(logit_t, dim=-1)
            f3b = float((prob_player - prob_t).square().mean())
            rows.append({"layer": layer, "test_window": w, "f1": f1, "f2": f2,
                         "f1_minus_f2": f1 - f2, "f3a": f3a, "f3b": f3b})
            print(f"L{layer:2d} w{w:2d}: F1={f1:.5f} F2={f2:.5f} F1-F2={f1-f2:+.5f} "
                  f"F3a={f3a:.4f} F3b={f3b:.6f}", flush=True)
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-measure", action="store_true", help="zero-API only (F4)")
    parser.add_argument("--date", default="2026-09-08")
    args = parser.parse_args()

    a23_cases = read_4b_cases(A23_DIR)
    assert len(a23_cases) == 72, f"expected 72 A23 cases, got {len(a23_cases)}"
    shas = {c["_sha"] for c in a23_cases}
    assert shas == {A23_LOCAL_SHA}, f"A23 sha mismatch: {shas}"
    f4_rows = [{"layer": c["layer"], "test_window": c["test_window"],
                "test_length": c["test_length"],
                "f4": 1.0 - c["gain"], "mse_standard": c["mse_standard"],
                "mse_player": c["mse_player"]} for c in a23_cases]
    a25_cases = read_4b_cases(A25_DIR) if (A25_DIR / "candidate-attention-shard0.json").exists() else []

    ledger = {
        "date": args.date,
        "side": "attention",
        "parent": {"name": "A23", "official": "14437 / 276s", "sha256": PARENT_SHA},
        "comparison": {"name": "A25", "official": "14057 / 254s OFFICIAL_REJECTED",
                       "f4_mean": (sum(1 - c["gain"] for c in a25_cases) / max(len(a25_cases), 1)) if a25_cases else None},
        "target": "final output gain >= 0.9 (i.e. F4 <= 0.1) on the 72-case 4B panel",
        "evaluator": "eval-v3 4B paired (archived JSON, zero API) + dense-cache recomputation",
        "cache": str(CACHE),
    }

    mean = lambda xs: sum(xs) / len(xs)
    ledger["F4"] = {
        "mean": mean([r["f4"] for r in f4_rows]),
        "per_layer": {str(L): mean([r["f4"] for r in f4_rows if r["layer"] == L]) for L in FA_LAYERS},
        "per_length": {str(n): mean([r["f4"] for r in f4_rows if r["test_length"] == n])
                       for n in sorted({r["test_length"] for r in f4_rows})},
        "coverage": "72/72 cases, zero API",
        "rows": f4_rows,
    }

    measure_rows = None
    if not args.skip_measure:
        mod = load(PARENT_SOLUTION, "a23_ledger")
        pack = torch.load(CACHE, map_location="cpu", weights_only=False)
        assert gpu_lock.acquire("A", "attention-error-ledger") == 0
        try:
            measure_rows = measure_parent(mod, pack)
        finally:
            gpu_lock.release("A", "attention-error-ledger")
        std_by_case = {(r["layer"], r["test_window"]): r["mse_standard"] for r in f4_rows}
        for r in measure_rows:
            std = std_by_case[(r["layer"], r["test_window"])]
            r["f1_share"] = r["f1"] / std
            r["f2_share"] = r["f2"] / std
            r["f1_minus_f2_share"] = r["f1_minus_f2"] / std
        ledger["F1"] = {
            "mean": mean([r["f1"] for r in measure_rows]),
            "mean_share_of_standard": mean([r["f1_share"] for r in measure_rows]),
            "per_layer_share": {str(L): mean([r["f1_share"] for r in measure_rows if r["layer"] == L])
                                for L in FA_LAYERS},
            "definition": "deploy Q/K attention output vs NVFP4 float reference, V fixed",
            "coverage": "72/72 cases, dense-cache recomputation",
        }
        ledger["F2"] = {
            "mean": mean([r["f2"] for r in measure_rows]),
            "mean_share_of_standard": mean([r["f2_share"] for r in measure_rows]),
            "per_layer_share": {str(L): mean([r["f2_share"] for r in measure_rows if r["layer"] == L])
                                for L in FA_LAYERS},
            "definition": "same T-coordinate float reference; pure code/scale allocation cost",
            "coverage": "72/72 cases",
        }
        ledger["F1_minus_F2"] = {
            "mean": mean([r["f1_minus_f2"] for r in measure_rows]),
            "mean_share_of_standard": mean([r["f1_minus_f2_share"] for r in measure_rows]),
            "definition": "transform-choice residual (center/GQA constraint, inverse alignment)",
        }
        ledger["F3a"] = {
            "mean": mean([r["f3a"] for r in measure_rows]),
            "per_layer": {str(L): mean([r["f3a"] for r in measure_rows if r["layer"] == L]) for L in FA_LAYERS},
            "definition": "logits MSE in T coordinates (unnormalized; logits space)",
        }
        ledger["F3b"] = {
            "mean": mean([r["f3b"] for r in measure_rows]),
            "per_layer": {str(L): mean([r["f3b"] for r in measure_rows if r["layer"] == L]) for L in FA_LAYERS},
            "definition": "post-softmax probability MSE (probability space, natural bound 1)",
        }
        ledger["F5"] = {
            "mean": ledger["F4"]["mean"] - ledger["F1"]["mean"],
            "definition": "F4 - F1: softmax nonlinearity coupling, length/layer distribution, V quantization",
        }
        ledger["rows"] = measure_rows + f4_rows

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_json = OUT_DIR / f"error_ledger_{args.date}.json"
    out_json.write_text(json.dumps(ledger, indent=2) + "\n", encoding="utf-8")

    lines = [
        f"# Attention error ledger {args.date}",
        "",
        f"- parent: A23 ({PARENT_SHA[:8]}..., official 14437/276s); comparison A25 14057/254s",
        f"- target: F4 <= 0.1 (final gain >= 0.9)",
        "",
        "| grid | mean | share of MSE_STD | note |",
        "|---|---|---|---|",
        f"| F4 final output | {ledger['F4']['mean']:.6f} | = itself | 72/72 zero API; prev run A25 {ledger['comparison']['f4_mean'] and round(ledger['comparison']['f4_mean'], 6)} |",
    ]
    if measure_rows:
        lines += [
            f"| F1 Q/K total | {ledger['F1']['mean']:.6f} | {ledger['F1']['mean_share_of_standard']:.4f} | 72/72 cache recomputation |",
            f"| F2 code/scale | {ledger['F2']['mean']:.6f} | {ledger['F2']['mean_share_of_standard']:.4f} | pure quantization cost |",
            f"| F1-F2 transform residual | {ledger['F1_minus_F2']['mean']:.6f} | {ledger['F1_minus_F2']['mean_share_of_standard']:.4f} | center/GQA/inverse alignment |",
            f"| F3a logits (abs) | {ledger['F3a']['mean']:.4f} | logits space | softmax absorbs most |",
            f"| F3b probability (abs) | {ledger['F3b']['mean']:.6f} | prob space | |",
            f"| F5 residual | {ledger['F5']['mean']:.6f} | | F4 - F1 (softmax/V/distribution) |",
            "",
            "## per-layer share (of mse_standard)",
            "",
            "| layer | F1 share | F2 share | F1-F2 share | F4 |",
            "|---|---|---|---|---|",
        ]
        for L in FA_LAYERS:
            lines.append(
                f"| L{L} | {mean([r['f1_share'] for r in measure_rows if r['layer']==L]):.4f} | "
                f"{mean([r['f2_share'] for r in measure_rows if r['layer']==L]):.4f} | "
                f"{mean([r['f1_minus_f2_share'] for r in measure_rows if r['layer']==L]):+.4f} | "
                f"{ledger['F4']['per_layer'][str(L)]:.4f} |")
    else:
        lines.append("| (F1/F2/F3 skipped: --skip-measure) | | | |")
    (OUT_DIR / f"error_ledger_{args.date}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nledger written: {out_json}")


if __name__ == "__main__":
    main()
