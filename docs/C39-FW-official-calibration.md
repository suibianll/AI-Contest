# C39-FW official calibration candidate

## Purpose

C39-FW is a small, single-mechanism candidate for separating the local
evaluator from the official evaluator after C38 showed a local/official
inversion. The parent is the official-compliant C21-C baseline, not C38.

Parent official anchor: C21-C, Linear `0.5311`, Attention `0.4497`, score
`14437`, time `166.6s`.

## Active algorithm

- Enable the existing full-64 Hessian weight refinement only when the Linear
  layer has input or output width at least `2048` (GPT-2 `fc`/`proj`).
- Use the wide-layer coverage `25%`, four scale-beam candidates, one
  coordinate pass, and the existing per-block full-H fallback.
- Keep q/k/v/o on the exact C21-C path.
- Keep activation settings at C21-C: refinement ratio `0.70`, 8-channel
  quadratic coverage `8%`, one sweep, and activation-only calibration gate.
- Keep cross-block refinement, R64, activation 16-channel refinement, sample
  importance, and hierarchy permutation disabled.

The full-64 objective uses the activation covariance `A^T A` with the weight
residual. It never constructs `A @ W` or a Linear output residual.

## Local result

GPT-2, 12 layers, sequence 128, calibration 2, test 2, amax6:

| Case | C21-C Linear | C39-FW Linear | Delta |
|---|---:|---:|---:|
| offset 0 | 0.5311 | 0.5357 | +0.46pp |
| offset 97 | 0.5148 | 0.5213 | +0.65pp |
| offset 193 | 0.5319 | 0.5385 | +0.66pp |
| offset 389 | 0.5235 | 0.5312 | +0.77pp |
| amax4, offset 0 | 0.4663 | 0.4740 | +0.77pp |
| pow2, offset 0 | 0.5454 | 0.5521 | +0.67pp |

Attention remains unchanged at approximately `0.4497` causal and `0.4942`
non-causal. CUDA algorithm-stage at offset 0 is `27.47s`; this is well below
the official 300-second limit, although only the official run is authoritative.

Candidate source SHA256:

`B8C9F2A4EB6553367DD17E73D30836AC8911DBEF33759FA8CF95E8C629317A71`

## Official result

The submitted C39-FW source received `14613` points in `159.2s`.

| Candidate | Official score | Official time | Delta vs C21-C |
|---|---:|---:|---:|
| C21-C | 14437 | 166.6s | baseline |
| C39-FW | 14613 | 159.2s | **+176 / -7.4s** |

This is a positive transfer from local to official evaluation. At local offset
0, the Linear gain was `+0.46pp`; the official gain is `+176` points. The
single pair is not sufficient to replace the historical score model, but it
confirms that the wide-only FULL64 mechanism is worth continuing.

The result also isolates the C38 failure: C39-FW keeps C21-C activation
settings and excludes narrow-layer FULL64, while C38 changed both at once and
scored `14092`. Therefore C38's negative result must be attributed to its
aggressive activation/narrow-layer combination or their interaction; it is not
evidence against the C39-FW wide-layer mechanism.

Decision: C39-FW is the current compliant official champion. Keep it frozen as
the parent for the next experiment.

## Official calibration rule

The official result is now available. Compare future candidates against both
the frozen C21-C anchor (`14437/166.6s`) and C39-FW (`14613/159.2s`):

- Official improvement positive, as in C39-FW: full-64 on FFN layers is a
  viable direction. Test coverage or solver changes one at a time.
- Official score near C21-C: local full-H improvement does not transfer;
  stop widening FULL64 and investigate official calibration-data/path
  differences.
- Official score below C21-C: FULL64 itself is incompatible with the official
  distribution; revert immediately to C21-C and do not combine this result
  with C38 activation changes.

Do not update the local-to-official score mapping using C38. C38 remains a
failed diagnostic candidate because its official score was `14092`, below the
C21-C score.
