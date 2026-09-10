"""Cross-check the two helper sets on identical inputs.

The candidate and the diagnostic produce identical codes given the same kernel,
yet opposite signs of result, so the divergence is either the kernel or these
helpers.  Each set is internally checked (the diagnostic's banded action was
verified against an explicit Toeplitz matrix) but they have NEVER been compared
to each other.

  diagnostic: apply_W (radius-padded unfold) + row_energy (index_add loop)
  candidate:  _vk_shifted (slice-add loop)  + _vk_row_energy

Same random delta, same random weights, same radius.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]


def main() -> int:
    torch.manual_seed(20260911)
    torch.set_grad_enabled(False)
    candidate = importlib.util.spec_from_file_location(
        "hx_candidate", HERE / "candidate" / "solution.py"
    )
    mod = importlib.util.module_from_spec(candidate)
    sys.modules["hx_candidate"] = mod
    candidate.loader.exec_module(mod)

    radius = mod._VK_RADIUS
    nbucket = mod._VK_NBUCKET
    nparam = mod._VK_NPARAM
    tokens = int(os.environ.get("HX_TOKENS", "37"))
    channels = 5

    delta = torch.randn(tokens, channels, dtype=torch.float64)
    weights = torch.randn(nparam, dtype=torch.float64)

    # --- the diagnostic's two helpers, reproduced exactly as vk3_rule.py has them
    def diag_apply(wg, d):
        near, far = wg[:nbucket], wg[nbucket]
        pad = torch.zeros(radius, d.shape[1], dtype=d.dtype)
        windows = torch.cat([pad, d, pad], dim=0).unfold(0, nbucket, 1)
        total = d.sum(0, keepdim=True)
        return torch.einsum("tcr,r->tc", windows, near) + far * (total - windows.sum(-1))

    def diag_energy(wg, n):
        energy = torch.zeros(n, dtype=torch.float64)
        counted = torch.zeros(n, dtype=torch.float64)
        for r in range(nbucket):
            off = r - radius
            lo, hi = max(0, -off), min(n, n - off)
            if hi <= lo:
                continue
            idx = torch.arange(lo, hi)
            energy.index_add_(0, idx, wg[r].square().expand(hi - lo))
            counted.index_add_(0, idx, torch.ones(hi - lo, dtype=torch.float64))
        return energy + (float(n) - counted) * wg[nbucket].square()

    print(f"radius={radius} nbucket={nbucket} nparam={nparam} tokens={tokens} channels={channels}")
    print()

    a = diag_apply(weights, delta)
    b = mod._vk_shifted(delta, weights, radius)
    gap_apply = float((a - b).abs().max())
    print(f"apply_W      vs _vk_shifted     max|gap| = {gap_apply:.3e}   "
          f"{'OK' if gap_apply < 1e-9 else '*** MISMATCH ***'}")

    ea = diag_energy(weights, tokens)
    eb = mod._vk_row_energy(weights, tokens, radius, torch.device("cpu"))
    gap_energy = float((ea - eb).abs().max())
    print(f"row_energy   vs _vk_row_energy  max|gap| = {gap_energy:.3e}   "
          f"{'OK' if gap_energy < 1e-9 else '*** MISMATCH ***'}")

    # the transpose helper the candidate builds inline in both places
    rev = torch.cat([weights[:nbucket].flip(0), weights[nbucket:]])
    ta = diag_apply(rev, delta)
    tb = mod._vk_shifted(delta, rev, radius)
    gap_t = float((ta - tb).abs().max())
    print(f"apply_W(rev) vs _vk_shifted(rev) max|gap| = {gap_t:.3e}   "
          f"{'OK' if gap_t < 1e-9 else '*** MISMATCH ***'}")

    print()
    if max(gap_apply, gap_energy, gap_t) < 1e-9:
        print("VERDICT: helpers agree on this input -- the divergence is in the KERNEL fit.")
        print("Next: compare the two fits bucket by bucket (totals / counts).")
    else:
        print("VERDICT: helpers DIVERGE -- this is the bug; fix the candidate's helper.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
