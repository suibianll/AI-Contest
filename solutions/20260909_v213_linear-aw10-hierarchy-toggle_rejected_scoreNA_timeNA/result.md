# v213 — output-aware lv3 hierarchy bit toggle

Status: `REJECTED` locally; official status `unregistered/NA`.

- Parent: retained v202 complete root, SHA256 `56dc805d6e5a3aef896db8021045740292735725d688b48e3d4393e55efcb2bd`.
- Candidate SHA256: `f43857621dcb89ba19163df7156efad4cc72ad892daf503cb6cc5a613b295595`.
- Mechanism: freeze the final dynamic `Q(A)`, select one natural 4-element group in each 64-block, and test the legal `lv3=1↔2` hierarchy toggle directly against the aggregate output residual. The group choice is shared across output rows; only `scale_lv3` changes.
- Fit: all calibration pairs and all rows supplied to the API; no parameter sweep or official-result wait.

## Local result

`eval-v3`, Qwen3.5-4B proxy-v2, Linear-only, target-side shard0 (56 cases):

| metric | v213 vs v202 |
|---|---:|
| mean delta gain | 0 |
| positive / negative / zero cases | 0 / 0 / 56 |
| candidate API total | 203.890s |
| parent API total | 179.254s |
| candidate / parent calibration API | 162.132s / 136.750s |

The legal hierarchy toggle produced no hard-output change and added calibration cost. The candidate does not replace v202. No official score or time is inferred.

