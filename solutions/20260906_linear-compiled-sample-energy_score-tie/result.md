# Linear compiled calibration sample-energy block order — OFFICIAL POSITIVE / LOCAL SCORE TIE

## Source and status

- Final implementation: `solution.py` (direct core calibration)
- Final SHA256: `d66128a62e7e068edc50c91f4d8e212f586a6edcaee5bea7d3564166e258b0f6`
- Parent: v189, official `17616/275s`
- Official result (user-reported): `17636/264s` (`+20` score, `-11s` vs v189)
- Local R3 decision: `REJECTED_SCORE_TIE`; the later official result is recorded here
- Root: unchanged v189; this archive is not promoted by this metadata update

The mechanism compiles one legal 64-channel block order during Linear weight
calibration. It averages final-transformed calibration-window block energy,
multiplies by the deployed-weight activation importance, and stores the
resulting order for the existing online GPTQ activation path. The codec,
weight transform, residual rank, Attention path, and API surface are unchanged.

## Proxy evidence

- R1 compact Linear: shard 0 mean delta `+0.001906528` (`44+/12-`), shard 1
  mean delta `+0.001949470` (`44+/12-`); no reasonableness blockers.
- R2 full Linear: 336 cases, mean delta `+0.002981296`, `292+/44-`, weighted
  L1 `0.0000357554`; compiled order reachable.
- OOD full Linear: mean delta `+0.003518763`; delta-gap change about
  `-0.000537467`, within the `0.01` diagnostic gate.

## Fresh default audit

The direct-core implementation produced:

- Linear `0.643867464427`
- Attention `0.752173407020`
- Overall `0.688994940507`
- API decomposition: `W=266.429245s`, `A=56.609574s`,
  `dyn_act=59.774616s`, `dyn_qkv=2.946691s`
- Official-time predictor: `279.445203s`

The score exactly tied the current local high `0.688994940507429`. Because
the local submission rule required a strict improvement, the local R3 decision
was `REJECTED_SCORE_TIE`. The user subsequently supplied an official result
of `17636/264s` for this archive; that later official positive result is
recorded separately from the original local gate decision.

The initial wrapper SHA is
`f6541fadbac21e2876000729cf6a87a0761c7f796ff5fd65365c752207758626`; its
predictor was `280.227124s`. The fused SHA is
`eaaac4b74f38e5711933d1b920ac641e0271aed073ab202b0c5a29ffe47e8572`; its
predictor was `281.100919s`. All three source variants and their fresh
reports are preserved in this archive.
