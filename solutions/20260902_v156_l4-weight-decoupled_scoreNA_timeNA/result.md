# v156 L4 Weight-decoupled stored-scale probe

- Status: `RETAINED` (official candidate pending; local evidence only so far)
- Parent: pre-A3 workbench parent; this snapshot was a formal single-file compilation of the probe
- Unique change: keep HiF4 sign/mantissa/lv2/lv3 fixed, solve one closed-form stored scale per
  expansive row/block under transformed calibration Gram, project to the nearest legal E6M2 code,
  and admit only when both folds' deployed output losses do not worsen. No dynamic search or extra
  API call was added.
- Source SHA256: `594EF2FBB70AE54E06BF2D896E11E637E4BA9AF67AD54C01F10D57136EB8DF85`

## Local evidence

The workbench implementation was evaluated on the `proxy-v2` Qwen effect panel (56 Linear + 5
Attention) against `l3-fc-parent`:

| scope | Linear | Attention | Overall | API total | wall |
|---|---:|---:|---:|---:|---:|
| effect | 0.588130853 | 0.757433277 | 0.602008101 | 203.994s | 216.749s |

The paired Linear delta was `+0.000107624` (5 improvements / 0 regressions / 51 unchanged), with
focus fc `+0.000376686` (5/0/11) and all q/k/v/o/proj controls plus Attention unchanged. W-only
was only `+0.000159`; this is a small proxy movement, not a meaningful official-score prediction.

On the identical 12-layer GPT-2 cache, the workbench moved Linear `0.519793773→0.519823226`
(`+0.000029454`) and `ffn_in/fc` `0.458206636→0.458383358` (`+0.000176722`); Attention stayed
`0.411099959`. This cross-model sign is positive but too small to justify a submission.

The formal effect rerun was intentionally stopped when the user requested immediate official
submission; the workbench effect and GPT-2 pair use the same single-file logic and remain the local
diagnostic evidence. No formal default JSON is claimed for this snapshot.

## Official fields and decision

- Official score: `unregistered`
- Official time: `NA`
- Official status: `unregistered`

This snapshot is retained specifically so the user can submit it to the official judge. The local
gain is small and must not be treated as evidence of an official improvement: official v86 is
`16744 / 222.7s`, while v147 is `16579 / 211s`, so the working hypothesis is a materially worse
v147 Linear path under the same v86-style Attention. Submit the exact file at
`solutions/20260902_v156_l4-weight-decoupled_scoreNA_timeNA/solution.py`; after the official score
and time return, either promote it or rename/archive it as rejected. Do not replace the root file
automatically.
