"""Recover the exact metric G from the parent h_inv state.

Parent:  h_inv = (G + c I)^{-1},  G = W_hat^T W_hat,  c = reg * mean(diag G).
Given h_inv and W_hat:  h_inv^{-1} - G = c I  =>  c = mean(diag(h_inv^{-1} - G)).
Then  G = h_inv^{-1} - c I  exactly (up to fp32 inversion error).
"""
import sys
from pathlib import Path
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "evaluator"))
import official_eval as v2  # noqa: E402

CACHE = ROOT / "artifacts/official_eval/cache/proxy-v3-calibration/56dc805d6e5a3aef-linear-452e6fe4aa5747590535.pt"
PACK = ROOT / "artifacts/official_eval/cache/qwen3.5-4b-proxy-v2.pt"
payload = torch.load(CACHE, map_location="cpu", mmap=True, weights_only=False)
pack = torch.load(PACK, map_location="cpu", mmap=True, weights_only=False)

for layer, role in ((0, "q"), (0, "o"), (2, "fc_gate")):
    entry = next(e for e in payload["weight_states"] if int(e["layer"]) == layer and str(e["role"]) == role)
    state, wparams = entry["state"], entry["params"]
    hinv32 = state["h_inv"].to(torch.float32)
    n = hinv32.shape[0]
    wshape = v2.dequantize_nvfp4(*v2._pair(pack["weights"][layer][role].to(torch.float32))).shape
    w_hat32 = v2.dequantize_hif4(dict(wparams), wshape).to(torch.float32)
    g_true32 = w_hat32.t().mm(w_hat32)
    hinv_inv32 = torch.cholesky_inverse(torch.linalg.cholesky(hinv32))
    d32 = hinv_inv32 - g_true32
    c32 = float(d32.diagonal().mean())
    off = d32 - c32 * torch.eye(n)
    g_rec32 = hinv_inv32 - c32 * torch.eye(n)
    # float64 reference for the same operations
    hinv64 = hinv32.to(torch.float64)
    g_true64 = w_hat32.to(torch.float64).t().mm(w_hat32.to(torch.float64))
    hinv_inv64 = torch.cholesky_inverse(torch.linalg.cholesky(hinv64))
    c64 = float((hinv_inv64 - g_true64).diagonal().mean())
    g_rec64 = hinv_inv64 - c64 * torch.eye(n)
    print(f"layer{layer}/{role} n={n}")
    print(f"  c(fp32)={c32:.6e}  c(fp64)={c64:.6e}  rel(c diff)={abs(c32-c64)/abs(c64):.2e}")
    print(f"  ||D - cI||/||G|| (fp32) = {float(off.norm()/g_true32.norm()):.3e}   "
          f"(should be ~0 if h_inv == (G+cI)^-1 exactly)")
    print(f"  rel_err G_rec(fp32) = {float((g_rec32-g_true32).norm()/g_true32.norm()):.3e}   "
          f"rel_err G_rec(fp64) = {float((g_rec64-g_true64).norm()/g_true64.norm()):.3e}")
    print(f"  cond(h_inv)={float(torch.linalg.cond(hinv64)):.3e}  cond(G)={float(torch.linalg.cond(g_true64)):.3e}")
