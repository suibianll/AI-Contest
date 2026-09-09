# v212 — 64-block shared integer code offset

Status: `REJECTED` locally; official status `unregistered/NA`.

- Parent: retained v202 complete root, SHA256 `56dc805d6e5a3aef896db8021045740292735725d688b48e3d4393e55efcb2bd`.
- Candidate SHA256: `143c459a11b76740ddbd229a9fb54cd2ebab156627f9e8de8e62b711ca38ab0c`.
- Mechanism: freeze the final dynamic `Q(A)`, select one natural 4-element group inside each 64-block, solve one code-space 4x4 normal equation, and share the resulting one-step integer offset across all output rows. Only mantissa/sign are changed; scale and hierarchy fields remain fixed.
- Fit: all calibration pairs and all rows supplied to the API; no parameter sweep or official-result wait.

## Local result

`eval-v3`, Qwen3.5-4B proxy-v2, Linear-only, target-side shard0 (56 cases; stopped after clear no-op):

| metric | v212 vs v202 |
|---|---:|
| mean delta gain | 0 |
| positive / negative / zero cases | 0 / 0 / 56 |
| candidate API total | 218.904s |
| parent API total | 43.797s |

No hard output changed, while calibration cost increased substantially. The first shard is sufficient to close this mechanism, so the remaining shards were not run.

The candidate does not replace v202. No official score or time is inferred.
