# v138 result

- Parent: v134
- Change: time-safe Attention path.  Candidate search is restricted to a small
  static reciprocal-balance/block-Hadamard/GQRB set, shortlist views are 128
  tokens, Attention calibration does not run the per-call Gram coordinate
  refiner, and dynamic Q/K/V return ordinary transformed HiF4 without Gram
  sweeps.  Linear v134 output-supervised cross64 is unchanged.
- Source SHA256: `3a120beb62443ff6a5bcdb89b5fad970ac6d8d45f48f40fe31812073060c2d10`
- Protocol: `official-shape-v1`, same read-only Qwen cache, CUDA

| Run | Linear mean | Attention mean | API total | Wall |
|---|---:|---:|---:|---:|
| first | 0.5073195049 | 0.7159419612 | 192.9958572 s | 216.3242606 s |
| idle rerun 2 | 0.5073195049 | 0.7159419612 | 187.9349367 s | 210.8548920 s |

The Linear score is identical to v134.  Attention is close to the official-pass
v86 local mean (`0.719696`) while its expensive dynamic Q/K/V work falls to
`3.71s` and Attention calibration to `36.25/36.79s`.  Official score/time are
not registered; local seconds are not an official-time guarantee.

Decision: promote v138 as the time-safe root and use it as the parent for the
next Linear precision experiments.
