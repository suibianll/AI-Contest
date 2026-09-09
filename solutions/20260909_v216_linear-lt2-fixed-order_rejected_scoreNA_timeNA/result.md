# v216 — fixed compiled activation GPTQ order

Status: `REJECTED` locally; official status `unregistered/NA`.

- Parent: retained v202 complete root, SHA256 `56dc805d6e5a3aef896db8021045740292735725d688b48e3d4393e55efcb2bd`.
- Candidate SHA256: `5ca016569a199abcdb782adf302267cee0079496fd7ba9e6ca38607849266ede`.
- Mechanism: use the sample-energy block order compiled in `activation_state` directly at runtime, removing the per-call activation-energy ranking; static calibration and weight state are unchanged.
- Fit: unchanged v202 calibration; no parameter sweep or official-result wait.

## Local result

`eval-v3`, Qwen3.5-4B proxy-v2, Linear-only, target-side shard0 (56 cases; stopped after clear regression):

| metric | v216 vs v202 |
|---|---:|
| mean delta gain | -0.003884 |
| median delta gain | -0.000748 |
| worst-20% tail delta | -0.004127 |
| positive / negative / zero cases | 13 / 43 / 0 |
| candidate API total | 171.883s |
| parent API total | 172.228s |
| candidate / parent calibration API | 129.929s / 129.912s |

The fixed compiled order regressed the hard activation output and did not produce a meaningful runtime reduction. The candidate does not replace v202. No official score or time is inferred.

