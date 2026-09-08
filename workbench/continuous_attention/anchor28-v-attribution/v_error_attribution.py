"""A28 pre-card diagnosis: V re-encode error attribution (zero API).

The V path re-encodes NVFP4 activations into HiF4 at deployment
(hif4_dynamic_quantize_v). F5-V = 0.275 mean share (L22 up to 0.66). User
unlocked the V code-allocation family on 2026-09-08. Before issuing a card,
quantify how much of the re-encode error is capturable at all:

  error_standard  : offsets=(), no refine  (pure amax/E6M2 path)
  error_baseline  : deployed state as-is   (5-value offset window + dynamic ratio)
  error_oracle    : wide offset grid + ratio 1.0 + blocks unbounded
  error_noimp     : baseline but importance=None (is head-level E[A^2] doing anything?)

Also reports plain MSE vs attention-mass-weighted MSE (A column mass from
calibration q/k) to expose weight-space vs output-space divergence.
"""
from pathlib import Path
import importlib.util
import json
import math
import sys
import time

import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "evaluator"))
sys.path.insert(0, str(ROOT / "workbench/v162_attention"))
import official_eval as ev  # noqa: E402
import gpu_lock  # noqa: E402

CACHE = ROOT / "artifacts/official_eval/cache/qwen3.5-4b-proxy-v2.pt"
FA_LAYERS = [0, 1, 5, 8, 15, 22]
WIDE_OFFSETS = (-6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7)
QN, KN, HD = 16, 4, 256


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def v_mse(mod, v_quant, v_scale, v_state):
    t0 = time.perf_counter()
    rec = mod._dequantize_hif4(
        mod.hif4_dynamic_quantize_v(v_quant, v_scale, KN, HD, v_state)
    ).float()
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    ref = mod._dequantize_nvfp4_float32(v_quant, v_scale).float()
    err = (rec - ref).square().mean().item()
    return err, dt, rec, ref


def clone_state(v_state, *, offsets=None, ratio=None, blocks=None, importance=None):
    s = dict(v_state)
    if offsets is not None:
        s["offsets"] = torch.tensor(offsets, dtype=torch.int8)
    if ratio is not None:
        s["max_refine_ratio"] = float(ratio)
    if blocks is not None:
        s["max_refine_blocks"] = int(blocks)
    if importance is not None:
        s["importance"] = importance
    return s


def main():
    mod = load(HERE / ".." / "anchor27-b" / "solution.py", "a27_sol_for_vdiag")
    pack = torch.load(CACHE, map_location="cpu", weights_only=False)
    assert gpu_lock.acquire("A", "anchor28-v-attribution") == 0
    rows = []
    try:
        for layer in FA_LAYERS:
            calib = [
                {role: tuple(t.cuda() for t in ev._pair(dense))
                 for role, dense in zip(("q", "k", "v"), pack["calibration_qkv"][f][layer])}
                for f in range(5)
            ]
            with torch.no_grad():
                state = mod._V189_CALIBRATION_ATTENTION(calib, QN, KN, HD)
                v_state = state["v_state"]
                imp = v_state["importance"]
                per_fold = []
                for f in range(5):
                    v_quant, v_scale = calib[f]["v"]
                    base_err, base_dt, _, ref = v_mse(mod, v_quant, v_scale, v_state)
                    std_state = clone_state(v_state, offsets=(), ratio=0.0)
                    std_err, _, _, _ = v_mse(mod, v_quant, v_scale, std_state)
                    oracle_state = clone_state(
                        v_state, offsets=WIDE_OFFSETS, ratio=1.0, blocks=None
                    )
                    orc_err, orc_dt, _, _ = v_mse(mod, v_quant, v_scale, oracle_state)
                    noimp_state = clone_state(
                        v_state, offsets=WIDE_OFFSETS, ratio=1.0, blocks=None,
                        importance=None,
                    )
                    noimp_err, _, _, _ = v_mse(mod, v_quant, v_scale, noimp_state)
                    # attention column mass from calibration q/k of this fold
                    # GQA: q (T,16,256) -> 4 groups x 4 subheads; k (T,4,256)
                    q = mod._dequantize_nvfp4_float32(*calib[f]["q"]).float()
                    k = mod._dequantize_nvfp4_float32(*calib[f]["k"]).float()
                    q4 = q.reshape(-1, QN, HD)
                    k4 = k.reshape(-1, KN, HD)
                    qg = q4.reshape(-1, KN, QN // KN, HD)
                    n_kv = k4.shape[0]
                    a_mass = torch.zeros(n_kv, device="cuda")
                    chunk = 256
                    sub = QN // KN
                    for i in range(0, qg.shape[0], chunk):
                        qc = qg[i:i + chunk]  # (c, G, sub, HD)
                        c = qc.shape[0]
                        logits = torch.einsum("cghd,thd->cght", qc, k4) / math.sqrt(HD)
                        q_idx = i + torch.arange(c, device="cuda")
                        kv_idx = torch.arange(n_kv, device="cuda")
                        causal = kv_idx[None, :] > q_idx[:, None, None, None]
                        logits = logits.masked_fill(causal, float("-inf"))
                        a = torch.softmax(logits, dim=-1)
                        a_mass += a.square().sum(dim=(0, 1, 2))
                    a_mass = a_mass / max(qg.shape[0], 1)
                    # recode error per token, weighted by A mass
                    rec_orc = mod._dequantize_hif4(
                        mod.hif4_dynamic_quantize_v(v_quant, v_scale, KN, HD, oracle_state)
                    ).float()
                    tok_err = (rec_orc - ref).square().mean(dim=1)
                    w = a_mass / a_mass.sum().clamp_min(1e-12)
                    weighted_err = float((tok_err * w).sum())
                    per_fold.append({
                        "fold": f, "tokens": int(v_quant.shape[0]),
                        "mse_standard": std_err, "mse_baseline": base_err,
                        "mse_oracle": orc_err, "mse_oracle_noimp": noimp_err,
                        "weighted_oracle": weighted_err,
                        "baseline_s": base_dt, "oracle_s": orc_dt,
                    })
                agg = {}
                for key in ("mse_standard", "mse_baseline", "mse_oracle",
                            "mse_oracle_noimp", "weighted_oracle"):
                    vals = [x[key] for x in per_fold]
                    agg[key] = sum(vals) / len(vals)
                rows.append({"layer": layer, "per_fold": per_fold, "mean": agg})
                print(f"L{layer:2d}: std={agg['mse_standard']:.3e} base={agg['mse_baseline']:.3e} "
                      f"oracle={agg['mse_oracle']:.3e} noimp={agg['mse_oracle_noimp']:.3e} "
                      f"woracle={agg['weighted_oracle']:.3e} "
                      f"oracle_gain={(1 - agg['mse_oracle'] / max(agg['mse_baseline'], 1e-30)) * 100:.2f}%",
                      flush=True)
    finally:
        gpu_lock.release("A", "anchor28-v-attribution")
    out = {"layers": rows, "wide_offsets": WIDE_OFFSETS,
           "verdict_note": "oracle_gain small => re-encode space exhausted"}
    (HERE / "v_error_attribution.json").write_text(
        json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print("saved v_error_attribution.json")


if __name__ == "__main__":
    main()
