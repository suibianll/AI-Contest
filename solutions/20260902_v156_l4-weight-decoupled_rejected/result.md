# v156 L4 Weight-decoupled stored-scale — rejected

- Status: `REJECTED` (official pass, but below v86)
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

- Official score: `16580` (user reported 2026-09-02; `-164` versus v86)
- Official time: `204.3s` (passes the strict `<300s` requirement)
- Official judge status: `pass`
- Archive decision: `REJECTED` because accuracy is below v86 `16744`

The official result rejects the mechanism: it saves `18.4s` versus v86 but loses `164` score points.
It is also one point below v155, so the small positive Qwen/GPT-2 proxy movements did not produce
an official accuracy gain. Keep this source only as negative evidence; do not replace root
`solution.py` and do not tune this stored-scale family further.
