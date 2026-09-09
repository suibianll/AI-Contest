# v211 — output-aware joint 4-code-group update

Status: `REJECTED` locally; official status `unregistered/NA`.

- Parent: retained v202 complete root, SHA256 `56dc805d6e5a3aef896db8021045740292735725d688b48e3d4393e55efcb2bd`.
- Candidate SHA256: `d61dfbb85a39334f61ca41cff5da10e0fde3cc984edb1ad4dcd453b00e3e75ef`.
- Mechanism: freeze the final dynamic `Q(A)`, solve one 4x4 output normal equation for one selected natural 4-element group per output row, round directly to the signed-mantissa integer lattice, and retain only exact calibration-loss reductions. Scale and hierarchy fields remain unchanged.
- Fit: all calibration pairs and all rows supplied to the API; no parameter sweep or official-result wait.

## Local result

`eval-v3`, Qwen3.5-4B proxy-v2, Linear-only, target-side shard0 (56 cases; stopped after clear rejection):

| metric | v211 vs v202 |
|---|---:|
| mean delta gain | -0.0171391319 |
| L1 delta gain | 0.018153 |
| worst-20% tail delta | -0.034169 |
| positive / negative / zero cases | 3 / 53 / 0 |

The candidate reached real hard-output code changes, but regressed broadly. Candidate API total was `236.264s` versus parent `62.697s`; local seconds are diagnostic only and are not an official-time claim. The first shard is sufficient to close this mechanism, so the remaining shards were not run.

The candidate does not replace v202. No official score or time is inferred.
