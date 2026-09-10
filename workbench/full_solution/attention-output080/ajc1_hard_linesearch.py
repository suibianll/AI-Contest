"""A-JC1: along the analytic direction, is there a scale at which the hard encode improves?

`ajc1_hard_gradient.py` found the deployed objective is flat within +-0.5 of the
parent point (8/8 finite differences exactly zero), while the analytic direction
does agree with the hard response once the step is large (cos 0.845 at delta 2.0).
The immediate question that follows is not whether the direction is right -- it
appears to be -- but whether any scale along it produces a hard MSE better than
the parent's.

This walks the analytic direction at fixed geometric scales and reports, for each,
the hard MSE and how many codes actually moved.  It is a diagnostic of the
direction-with-scale, not a parameter scan for a card: the scales are fixed here,
nothing is tuned, and no candidate is produced.

If some scale beats the parent, the direction is usable and the card's problem is
picking the scale (a solve-side fix).  If every scale is worse, the direction is
right but the staircase has no step in a good direction -- a different, harder
result.

CPU only, synthetic contract setting, no 4B, no shard, no candidate.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


smoke = load("ajc1_smoke_ls", HERE / "contract_smoke.py")
ref = smoke.ref

SCALES = (0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0)


def main() -> int:
    torch.set_num_threads(2)
    torch.manual_seed(0)

    q, k, v = torch.randn(8, 128), torch.randn(8, 64), torch.randn(8, 64)
    pq, pk, pv = [ref.encode_standard_hif4(x) for x in (q, k, v)]
    q0, k0, v0 = [ref.decode_standard_hif4(p).double() for p in (pq, pk, pv)]
    fq, bq = smoke.features(q, pq)
    fk, bk = smoke.features(k, pk)
    zero = torch.zeros(8, dtype=torch.float64)
    target = smoke.attention(q.double(), k.double(), v.double())

    def relaxed(t):
        return smoke.attention(q0 + bq @ t[:4], k0 + bk @ t[4:], v0)

    def encode(t):
        return smoke.hard(pq, fq, t[:4]), smoke.hard(pk, fk, t[4:])

    def hard_mse(t):
        aq, ak = encode(t)
        out = smoke.attention(
            ref.decode_standard_hif4(aq).double(), ref.decode_standard_hif4(ak).double(), v0
        )
        return float((out - target).square().mean())

    def changed(t):
        aq, ak = encode(t)
        return int((aq["mant"] != pq["mant"]).sum()), int((ak["mant"] != pk["mant"]).sum())

    residual = (relaxed(zero) - target).flatten()
    jac = torch.autograd.functional.jacobian(relaxed, zero).reshape(-1, 8)
    h = jac.T @ jac / residual.numel()
    g = jac.T @ residual / residual.numel()
    ridge = max(1e-4 * float(h.trace()) / 8, 1e-12)
    theta = torch.linalg.solve(h + ridge * torch.eye(8, dtype=h.dtype), -g)

    parent = hard_mse(zero)
    print(f"parent hard MSE = {parent:.10f}")
    print(f"{'scale':>7} {'hard MSE':>14} {'vs parent':>11} {'changed q/k':>13} {'smooth pred':>14}")
    rows = []
    for scale in SCALES:
        t = theta * scale
        value = hard_mse(t)
        cq, ck = changed(t)
        smooth = float((residual + jac @ t).square().mean())
        rows.append(
            {
                "scale": scale,
                "hard_mse": value,
                "delta_pct": 100.0 * (value - parent) / parent,
                "changed_q": cq,
                "changed_k": ck,
                "smooth_pred_mse": smooth,
            }
        )
        print(
            f"{scale:>7.2f} {value:>14.10f} {100.0*(value-parent)/parent:>10.3f}% "
            f"{cq:>6}/{ck:<6} {smooth:>14.10f}"
        )

    best = min(rows, key=lambda r: r["hard_mse"])
    print()
    print(
        f"best scale {best['scale']:.2f}: hard MSE {best['hard_mse']:.10f} "
        f"({best['delta_pct']:+.3f}% vs parent), changed q/k = {best['changed_q']}/{best['changed_k']}"
    )
    print(
        f"any scale better than the parent: "
        f"{any(r['delta_pct'] < 0 for r in rows if r['scale'] > 0)}"
    )
    print(
        f"scales with zero code change: "
        f"{[r['scale'] for r in rows if r['changed_q'] == 0 and r['changed_k'] == 0]}"
    )

    (HERE / "ajc1-line-search.json").write_text(
        json.dumps(
            {
                "scope": "synthetic CPU contract only; not 4B or official evidence",
                "parent_hard_mse": parent,
                "theta": theta.tolist(),
                "rows": rows,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"\nwrote {HERE / 'ajc1-line-search.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
