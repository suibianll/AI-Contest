# v215 — output-aware E6M2 scale-factor adjacent step

Status: `REJECTED` locally; official status `unregistered/NA`.

- Parent: retained v202 complete root, SHA256 `56dc805d6e5a3aef896db8021045740292735725d688b48e3d4393e55efcb2bd`.
- Candidate SHA256: `1341424bcb4e30123be7ec1413e97d62c9c75a9a4f736f5b28498214dd49cc3a`.
- Mechanism: freeze the final dynamic `Q(A)`; for each output row and natural 64-block, use the residual directional term to select one adjacent E6M2 code step, then accept only the exact aggregate product-loss decrease. Hierarchy and mantissa/sign fields remain fixed.
- Fit: all calibration pairs and all rows supplied to the API; no parameter sweep or official-result wait.

## Local result

`eval-v3`, Qwen3.5-4B proxy-v2, Linear-only, target-side shard0 (56 cases; stopped after clear regression):

| metric | v215 vs v202 |
|---|---:|
| mean delta gain | -0.150813 |
| median delta gain | -0.100438 |
| worst-20% tail delta | -0.263206 |
| positive / negative / zero cases | 0 / 56 / 0 |
| candidate API total | 203.182s |
| parent API total | 172.276s |
| candidate / parent calibration API | 161.955s / 129.763s |

The fixed adjacent scale-factor step regressed every shard0 case, with the largest role-family regression on `proj` (`-0.412170`). The candidate does not replace v202. No official score or time is inferred.
