# v218 — C76.1 fixed Q-only headwise range permutation

Status: `REJECTED` locally; official status `unregistered/NA`.

- Parent: retained v202 complete root, SHA256 `56dc805d6e5a3aef896db8021045740292735725d688b48e3d4393e55efcb2bd`.
- Candidate SHA256: `cf71083b146aa78639550160c67c7e1b96e14817c70bdcba4cbd83600ecfdd05`.
- Mechanism: enable only the first independent Q-only headwise range-permutation candidate in the A1 output selector; candidate cap is fixed at one, while K, V, and dynamic APIs are unchanged.
- Fit: unchanged v202/v195 attention calibration inputs; no permutation sweep and no official-result wait.

## Local result

`eval-v3`, Qwen3.5-4B proxy-v2, Attention-only, target-side shard0 (12 cases):

| metric | v218 vs v202 |
|---|---:|
| mean delta gain | +0.000000 |
| median delta gain | +0.000000 |
| worst-20% tail delta | +0.000000 |
| positive / negative / zero cases | 0 / 0 / 12 |
| candidate API total | 5.239s |
| parent API total | 5.471s |
| candidate / parent calibration API | 4.901s / 5.124s |

The fixed Q-only permutation did not change any hard output, so it provides no evidence of an actionable improvement and does not replace v202. No official score or time is inferred.
