"""A-JC1, the check that was never done: does the analytic parameter direction match the hard response?

The corrected report (report-080-verdict.md section 4) says the A-JC1 verdict must
rest on three things that were never measured.  This script does the first and
second of them:

  1. the parameter direction  g_theta = J_theta^T r  -- already computed by the
     contract, but never compared against anything;
  2. the actual hard response  d(hard MSE)/d(theta)  -- measured here by finite
     differences through the same `hard()` the candidate would deploy.

The comparison is the point.  `hard()` rounds: `(phi @ theta).round()` maps a
continuous direction onto integer code steps, so the hard objective is a staircase
in theta -- flat almost everywhere, with jumps where a code crosses.  If that is
what the data shows, then the solve is minimising a smooth surrogate whose
gradient does not exist in the deployed objective, and no amount of solver quality
fixes it; if the hard finite differences instead align with the analytic g, the
surrogate is a fair stand-in and the mismatch must be sought elsewhere.

This is the same discipline the session's earlier mistakes were caught by: measure
the quantity the decision actually needs, not a nearby one that is easier.

CPU only, synthetic setting of the contract; no 4B, no shard, no candidate.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


smoke = load("ajc1_smoke", HERE / "contract_smoke.py")
ref = smoke.ref


def cosine(a, b):
    a = a.reshape(-1).double()
    b = b.reshape(-1).double()
    return float((a @ b) / (a.norm() * b.norm()).clamp_min(1e-30))


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

    def hard_mse(t):
        aq = smoke.hard(pq, fq, t[:4])
        ak = smoke.hard(pk, fk, t[4:])
        out = smoke.attention(
            ref.decode_standard_hif4(aq).double(), ref.decode_standard_hif4(ak).double(), v0
        )
        return float((out - target).square().mean())

    residual = (relaxed(zero) - target).flatten()
    jac = torch.autograd.functional.jacobian(relaxed, zero).reshape(-1, 8)
    h = jac.T @ jac / residual.numel()
    g = jac.T @ residual / residual.numel()
    ridge = max(1e-4 * float(h.trace()) / 8, 1e-12)
    theta = torch.linalg.solve(h + ridge * torch.eye(8, dtype=h.dtype), -g)

    parent_hard = hard_mse(zero)
    solved_hard = hard_mse(theta)

    # Hard finite differences, per parameter, at theta = 0 and at the solved theta.
    def hard_fd(base, delta):
        entries = []
        for index in range(8):
            up = base.clone()
            down = base.clone()
            up[index] += delta
            down[index] -= delta
            entries.append((hard_mse(up) - hard_mse(down)) / (2.0 * delta))
        return torch.tensor(entries, dtype=torch.float64)

    report = {}
    for delta in (0.25, 0.5, 1.0, 2.0):
        fd0 = hard_fd(zero, delta)
        fds = hard_fd(theta, delta)
        report[f"delta={delta}"] = {
            "hard_fd_at_zero": fd0.tolist(),
            "zeros_at_zero": int((fd0 == 0).sum()),
            "cos_to_analytic_g_at_zero": cosine(fd0, g),
            "hard_fd_at_solved": fds.tolist(),
            "zeros_at_solved": int((fds == 0).sum()),
        }
        print(
            f"[delta={delta}] hard FD at theta=0: {fd0.norm():.3e} "
            f"(exactly zero in {int((fd0==0).sum())}/8 entries, "
            f"cos to analytic g = {cosine(fd0, g):+.3f}) | "
            f"at solved theta: norm {fds.norm():.3e}, "
            f"exact zeros {int((fds==0).sum())}/8",
            flush=True,
        )

    print()
    print(f"analytic g (8)   = {[round(x, 6) for x in g.tolist()]}")
    print(f"hard FD at 0, d=0.5 = {[round(x, 6) for x in report['delta=0.5']['hard_fd_at_zero']]}")
    print()
    print(f"hard MSE: parent(theta=0) {parent_hard:.10f} | solved theta {solved_hard:.10f} "
          f"({'worse' if solved_hard > parent_hard else 'better'} by "
          f"{100*(solved_hard-parent_hard)/parent_hard:+.2f}%)")

    payload = {
        "scope": "synthetic CPU contract only; not 4B or official evidence",
        "question": (
            "Does the analytic parameter direction g_theta = J^T r agree with the "
            "hard finite-difference response of the deployed objective?"
        ),
        "analytic_g": g.tolist(),
        "parent_hard_mse": parent_hard,
        "solved_hard_mse": solved_hard,
        "theta": theta.tolist(),
        "by_delta": report,
    }
    (HERE / "ajc1-hard-gradient.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    print(f"\nwrote {HERE / 'ajc1-hard-gradient.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
