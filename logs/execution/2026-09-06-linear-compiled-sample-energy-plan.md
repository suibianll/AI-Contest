# 2026-09-06 Linear 编译校准样本能量块序执行记录

## R0 / R1

Parent was root v189, source SHA
`261202248a0146a2ee45f3df60bd1979bb8171b7c162921013b0024c848617af`.
The candidate used the shared proxy-v2 cache and CUDA with `eval-v3`.
Compact shard 0 returned mean delta `+0.001906528`, median
`+0.001085706`, and `44+/12-`; shard 1 returned mean delta `+0.001949470`,
median `+0.000858329`, and `44+/12-`. There were no reasonableness issues.
The compiled order was reachable and contained legal 64-channel blocks.

## R2

The fixed six-shard Linear run returned 336 paired cases with mean delta
`+0.002981296`, `292+/44-`, and weighted L1 `0.0000357554`. The full OOD
run returned mean delta `+0.003518763`; the in-distribution minus OOD
delta-gap change was approximately `-0.000537467`, within the diagnostic
gate. No interface, finite-output, or control blocker was observed.

## R3 fresh default

Three source variants were audited on the same cache. All had the same local
score: Linear `0.643867464427`, Attention `0.752173407020`, Overall
`0.688994940507`.

| variant | SHA prefix | predictor | decision |
| --- | --- | ---: | --- |
| initial wrapper | `f6541fad` | `280.227124s` | time reject |
| fused calibration | `eaaac4b7` | `281.100919s` | time reject |
| direct core | `d66128a6` | `279.445203s` | score tie |

The direct-core version passed the time predictor, but its Overall exactly
tied the local high `0.688994940507429`. Under the strict-improvement rule it
was not submitted. The root remains v189. Complete source and evidence are
archived at
`solutions/20260906_linear-compiled-sample-energy_score-tie/`.
