# v157 exact-v86 + ROAB-P2 — rejected

- Status: `REJECTED` (official pass, but 15 points below v86)
- Parent: exact v86 single-file archive, official `16744 / 222.7s`
- Parent SHA256: `E7A16D6991DBB70A593FBE87D0C5D1D8FD38F801665354A01FFAF2F0A96F03CD`
- Unique change: after all v86 Linear transforms are frozen, learn one bounded analytic 2x2
  reciprocal pair transform. Activations use `X @ U`; weights use `W @ U^{-T}`. A bounded plain
  HiF4 output score chooses between the unchanged v86 coordinate and this one proposal.
- Frozen path: v86 Attention, Weight/Activation codecs, existing Linear transforms and online
  refinement logic are unchanged. There is no token/sequence search in the dynamic path.
- Source SHA256: `984BF752156187B8892894060A99FE52027E2457F37FC23C11657041B29B86E1`

## Why this experiment exists

The official evidence supports only one clean positive Linear increment: v138 and v140 share the
same reduced Attention and v140 adds ROAB-P2, moving the official score from `15715` to `15838`
(`+123`) while remaining below 300 seconds (`207s`). The absolute v140 result is rejected because
its larger Linear/Attention bundle is worse than v86; it does not show that the isolated ROAB
increment is negative. v157 tests that increment directly on the reproducible official v86 parent.

## Verification evidence

No local ranking panel was run. Local work was deliberately limited to implementation checks:

- `python -m py_compile solutions/20260902_v157_v86-roab-only_rejected/solution.py`
- `python -m pytest -q` -> `35 passed in 8.42s`
- analytic invariant check: max absolute error of `XW^T - (XU)(WU^{-T})^T` was
  `9.5367431640625e-06`; transformed covariance error was `3.814697265625e-06`
- public-API smoke on deterministic synthetic CPU tensors: all six outputs/state passed evaluator
  validation; Linear calibration was `0.183s`, Attention calibration was `0.270s`
- forced selected-ROAB branch: state shape `(channels/2, 2, 2)` and both weight/dynamic activation
  outputs passed evaluator validation
- forced rejection branch: Linear calibration and dynamic activation were field-for-field equal to
  exact v86; Attention calibration and Q/K/V dynamic outputs were also field-for-field equal
- isolated single-file import outside the archive directory exposed all six required APIs

These checks establish legality, coordinate consistency and parent fallback only. They are not an
official-score proxy and provide no local Linear/Attention mean.

## Evaluation fields

- Local protocol: deterministic synthetic legality/invariance smoke only
- Model/data revision: `NA` (no model panel was run)
- Cache: `NA`
- Device: CPU
- Local Linear mean: `NA`
- Local Attention mean: `NA`
- Local API total/wall: `NA` (the two synthetic calibration timings above are not comparable)
- Official score: `16729` (user reported 2026-09-02; `-15` versus v86)
- Official time: `218.96s` (`-3.74s` versus v86; passes strict `<300s`)
- Official judge status: `pass`
- Archive decision: `REJECTED` because accuracy is below v86 `16744`

Decision: reject and keep only as attribution evidence. The `v138→v140 +123` ROAB increment does
not transfer when stacked onto exact v86; the effect was context-dependent rather than a portable
ROAB main effect. Do not replace root `solution.py` or tune ROAB thresholds/pair sizes further.
