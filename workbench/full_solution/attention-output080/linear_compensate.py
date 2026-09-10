"""Linear: X can compensate W's error, and W is compiled into the state.

The output error is `dX W_hat^T + X_ref dW^T` -- the two errors interact, and X's
can cancel W's.  The least-squares X that makes the product exact is

    X* = X_ref W_ref^T (W_hat^T)^+

whose operator `W_ref^T (W_hat^T)^+` is (in, in) and can be compiled into the
activation state, so the dynamic API can form X* from its own input alone.  Unlike
Attention, where the other operand's *input* is out of reach, here the other
operand is a *weight* already fixed at calibration.

This measures the headroom: encode X* with the same standard HiF4 encoder and score
the actual product.  It is a ceiling probe -- X* is not a code change of X_ref, so
realising it means an input transform, which the root already does elsewhere
(smooth/permutation/rotation).

CPU only, no shard, no candidate.
"""
from __future__ import annotations
import importlib.util, json, statistics, sys
from pathlib import Path
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]

def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec); sys.modules[name] = m
    spec.loader.exec_module(m); return m

def main():
    torch.set_num_threads(2); torch.set_grad_enabled(False)
    sol = load("lc_sol", ROOT / "solution.py")
    sys.path.insert(0, str(ROOT / "evaluator"))
    import official_eval as v2
    pack = torch.load(ROOT / "artifacts/official_eval/cache/qwen3.5-4b-proxy-v2.pt",
                      map_location="cpu", mmap=True, weights_only=False)
    dev = torch.device("cpu"); layer = 0; rows = []
    for role in ("q", "o"):
        calib = [tuple(t.to(dev) for t in v2._pair(pack["calibration_activations"][role][s][layer].to(torch.float32))) for s in range(2)]
        wp = [t.to(dev) for t in v2._pair(pack["weights"][layer][role].to(torch.float32))]
        cal = sol.hif4_calibration_and_quantize_weight(wp[0], wp[1], calib)
        state = cal["activation_state"]
        shape = tuple(pack["weights"][layer][role].shape)
        w_hat = v2.dequantize_hif4(v2._cpu_params(cal["weight_params"]), shape).to(torch.float32)
        w_ref = v2.dequantize_nvfp4(*wp).to(torch.float32)
        w_std = v2.decode_standard_hif4(v2.encode_standard_hif4(w_ref)).to(torch.float32)
        op = w_ref.T @ torch.linalg.pinv(w_hat.T)      # (in, in), compiled operator
        for window in range(len(pack["test_activations"][role])):
            t = pack["test_activations"][role][window][layer]
            if t is None or t.shape[0] > 256: continue
            pr = v2._pair(t.to(torch.float32))
            x_ref = v2.dequantize_nvfp4(*[q.to(dev) for q in pr]).to(torch.float32)
            x_std = v2.decode_standard_hif4(v2.encode_standard_hif4(x_ref)).to(torch.float32)
            x_pr = sol.hif4_dynamic_quantize_activation(pr[0].to(dev), pr[1].to(dev), dict(state))
            x_hat = v2.dequantize_hif4(v2._cpu_params(x_pr), x_ref.shape).to(torch.float32)
            o_ref = x_ref @ w_ref.T
            mse_std = float((x_std @ w_std.T - o_ref).square().mean())
            mse_cur = float((x_hat @ w_hat.T - o_ref).square().mean())
            star = x_ref @ op                            # the least-squares X
            x_star = v2.decode_standard_hif4(v2.encode_standard_hif4(star)).to(torch.float32)
            mse_star = float((x_star @ w_hat.T - o_ref).square().mean())
            rows.append({"role": role, "window": window,
                         "gain_current": 1.0 - mse_cur/mse_std,
                         "gain_compensated": 1.0 - mse_star/mse_std})
    print(f"cases={len(rows)}")
    for role in ("q", "o"):
        sub = [r for r in rows if r["role"] == role]
        if not sub: continue
        c = statistics.mean(r["gain_current"] for r in sub)
        s = statistics.mean(r["gain_compensated"] for r in sub)
        print(f"  {role}: gain {c:.4f} -> {s:.4f} ({s-c:+.4f})  n={len(sub)}")
    (HERE/"linear-compensate.json").write_text(json.dumps({"rows": rows}, indent=2)+"\n", encoding="utf-8")
    print(f"wrote {HERE/'linear-compensate.json'}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
