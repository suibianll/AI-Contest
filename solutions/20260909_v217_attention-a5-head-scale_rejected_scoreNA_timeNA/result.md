# v217 — A5 fixed reciprocal temperature 1.25

Status: `REJECTED` locally; official status `unregistered/NA`.

- Parent: retained v202 complete root, SHA256 `56dc805d6e5a3aef896db8021045740292735725d688b48e3d4393e55efcb2bd`.
- Candidate SHA256: `62d7474d090275a47d0e250570aa1ed5207c1f889b9932db21f8426454737a19`.
- Mechanism: enable exactly one reciprocal Q/K temperature factor, `1.25`, in the existing A1 output-aware selector; V and dynamic APIs are unchanged.
- Fit: unchanged v202/v195 attention calibration inputs; no factor sweep and no official-result wait.

## Local result

`eval-v3`, Qwen3.5-4B proxy-v2, Attention-only, target-side shard0 (12 cases):

| metric | v217 vs v202 |
|---|---:|
| mean delta gain | +0.000000 |
| median delta gain | +0.000000 |
| worst-20% tail delta | +0.000000 |
| positive / negative / zero cases | 0 / 0 / 12 |
| candidate API total | 5.301s |
| parent API total | 5.711s |
| candidate / parent calibration API | 4.938s / 5.341s |

The fixed factor did not change any hard output, so it provides no evidence of an actionable improvement and does not replace v202. No official score or time is inferred.
