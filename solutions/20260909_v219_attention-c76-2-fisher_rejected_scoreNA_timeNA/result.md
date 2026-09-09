# v219 — C76.2 fixed output-Fisher Q/K importance

Status: `REJECTED` locally; official status `unregistered/NA`.

- Parent: retained v202 complete root, SHA256 `56dc805d6e5a3aef896db8021045740292735725d688b48e3d4393e55efcb2bd`.
- Candidate SHA256: `b2d2102c00ea29912938c8c463b732b418d5cf2ae52fe87ebd1379a9b07a28eb`.
- Mechanism: enable calibration-only output-Fisher Q/K importance with one fixed blend `0.5`; the legal state format, V, and dynamic APIs are unchanged.
- Fit: unchanged v202/v195 attention calibration inputs; no blend or role sweep and no official-result wait.

## Local result

`eval-v3`, Qwen3.5-4B proxy-v2, Attention-only, target-side shard0 (12 cases):

| metric | v219 vs v202 |
|---|---:|
| mean delta gain | +0.000000 |
| median delta gain | +0.000000 |
| worst-20% tail delta | +0.000000 |
| positive / negative / zero cases | 0 / 0 / 12 |
| candidate API total | 5.429s |
| parent API total | 5.531s |
| candidate / parent calibration API | 5.082s / 5.172s |

The fixed output-Fisher importance did not change any hard output, so it provides no evidence of an actionable improvement and does not replace v202. No official score or time is inferred.
