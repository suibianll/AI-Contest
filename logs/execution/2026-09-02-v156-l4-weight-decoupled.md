# v156 L4-WD candidate and submission record

## Status

`RETAINED / OFFICIAL CANDIDATE PENDING`. This is a single-file candidate for user submission; no
official score or runtime has been inferred. The root `solution.py` is unchanged.

Source: `solutions/20260902_v156_l4-weight-decoupled_scoreNA_timeNA/solution.py`
SHA256: `594EF2FBB70AE54E06BF2D896E11E637E4BA9AF67AD54C01F10D57136EB8DF85`

## Mechanism

Starting from the pre-A3 parent, keep the existing BOAT/ROAB and final weight codes fixed. For
expansive Linear shapes only, compute a transformed calibration block Gram, solve one closed-form
stored scale for each row/64-channel block, project it to the nearest legal E6M2 code, and admit the
whole update only when both calibration folds' actual deployed output losses do not worsen. The
dynamic APIs and Attention path are unchanged; this is not v155 permutation stacking and adds no
online search.

## Evidence

Qwen `proxy-v2` effect panel (56 Linear + 5 Attention), workbench implementation:

| Linear | Attention | Overall | API total | wall | paired Linear delta |
|---:|---:|---:|---:|---:|---:|
| 0.588130853 | 0.757433277 | 0.602008101 | 203.994s | 216.749s | +0.000107624 (5/0/51) |

Focus `fc` delta is `+0.000376686` (5/0/11); q/k/v/o/proj controls and Attention are no-op. The
evaluator decomposition shows W-only `+0.000159`, A-only `−0.002694`, Both `+0.000108`, with a
small positive interaction; this is a weak signal, not an official-score proxy.

Strict GPT-2 paired cache check:

| candidate | Linear | Attention | Overall | API total | wall |
|---|---:|---:|---:|---:|---:|
| pre-A3 parent | 0.519793773 | 0.411099959 | 0.470387493 | 76.194s | 83.974s |
| v156 | 0.519823226 | 0.411099959 | 0.470403559 | 82.579s | 91.277s |
| delta | +0.000029454 | 0 | +0.000016066 | +6.386s | +7.303s |

The formal Qwen effect rerun was stopped when the user requested immediate official submission; no
partial JSON is treated as a result. Workbench evidence is retained in
`artifacts/official_eval/l4-weight-decoupled-effect.json` and
`artifacts/official_eval/gpt2-l4-weight-decoupled.json`.

## Submission handoff

Submit the exact single file at
`solutions/20260902_v156_l4-weight-decoupled_scoreNA_timeNA/solution.py`. After the official score
and time return, update `result.md`, `docs/current-solution-status.md`, and `solutions/README.md`;
if it is below v86 or times out, rename the directory with `_rejected`/`_timeout` according to the
archive rules. Until then, official fields remain `unregistered/NA`.
