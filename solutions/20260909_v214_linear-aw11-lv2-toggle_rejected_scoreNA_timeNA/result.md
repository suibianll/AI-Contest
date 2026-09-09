# v214 — output-aware lv2 hierarchy bit toggle

Status: `REJECTED` locally; official status `unregistered/NA`.

- Parent: retained v202 complete root, SHA256 `56dc805d6e5a3aef896db8021045740292735725d688b48e3d4393e55efcb2bd`.
- Candidate SHA256: `b3f79cea6d91c02857d68c919f30c5351eae51e3e72a5a22a252ee63d70030de`.
- Mechanism: freeze the final dynamic `Q(A)`, select one natural 8-element group in each 64-block, and test the legal `lv2=1↔2` hierarchy toggle directly against the aggregate output residual. The group choice is shared across output rows; only `scale_lv2` changes.
- Fit: all calibration pairs and all rows supplied to the API; no parameter sweep or official-result wait.

## Local result

`eval-v3`, Qwen3.5-4B proxy-v2, Linear-only, target-side shard0 (56 cases):

| metric | v214 vs v202 |
|---|---:|
| mean delta gain | 0 |
| positive / negative / zero cases | 0 / 0 / 56 |
| candidate API total | 202.267s |
| parent API total | 172.438s |
| candidate / parent calibration API | 161.157s / 130.441s |

The legal hierarchy toggle produced no hard-output change and added calibration cost. The candidate does not replace v202. No official score or time is inferred.

