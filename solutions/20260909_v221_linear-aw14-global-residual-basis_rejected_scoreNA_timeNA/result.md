# v221 — L-AW14 shared output-residual basis A@W fit

Status: `REJECTED` locally; official status `unregistered/NA`.

- Parent: retained v202 complete root, SHA256 `56dc805d6e5a3aef896db8021045740292735725d688b48e3d4393e55efcb2bd`.
- Candidate SHA256: `01fa4d8c72e2f584562c746543af989a65bd3a0c25c6eebccb92b0969eb4f5c4`.
- Mechanism: form one fixed rank-8 right singular output basis from the complete calibration residual, reuse it for every natural 64-column block, solve sequential block coefficients, and accept only exact full-output residual reductions after legal HiF4 projection.
- Fit: all calibration pairs and rows supplied by eval-v3; no rank, basis, projection, or window sweep and no official-result wait.

## Local result

`eval-v3`, Qwen3.5-4B proxy-v2, Linear-only target-side shard0 (56 cases):

| metric | v221 vs v202 |
|---|---:|
| mean delta gain | -0.032567977 |
| median delta gain | -0.019347 |
| worst-20% tail delta | -0.039451 |
| positive / negative / zero cases | 0 / 56 / 0 |
| candidate API total | 203.721s |
| parent API total | 171.851s |
| candidate calibration API | 162.410s |
| parent calibration API | 130.146s |

The shared rank-8 basis was constructed and the hard output changed, but every paired Linear case regressed, with the strongest regression in the `o` role. This concrete implementation is closed as `REJECTED` and does not replace v202. No official score or time is inferred.
