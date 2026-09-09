# v220 — L-AW13 zero-to-smallest signed-code insertion

Status: `REJECTED` locally; official status `unregistered/NA`.

- Parent: retained v202 complete root, SHA256 `56dc805d6e5a3aef896db8021045740292735725d688b48e3d4393e55efcb2bd`.
- Candidate SHA256: `9c7f4648530b3628c25b8e00e169aa32eb5dcc19dc772dd9cfb7a7cd30cf1fbc`.
- Mechanism: for each natural 64-value block, select one zero-code weight element by frozen output leverage, try the two legal signs of mantissa `0.25`, and accept the exact aggregate calibration product-loss reduction. The old LC2 implementation did not execute this zero-to-negative-code direction.
- Fit: all calibration pairs and rows supplied by eval-v3; no parameter sweep, no official-result wait.

## Local result

`eval-v3`, Qwen3.5-4B proxy-v2, Linear-only target-side shard0 (56 cases):

| metric | v220 vs v202 |
|---|---:|
| mean delta gain | -0.000051551 |
| median delta gain | -0.000073392 |
| worst-20% tail delta | -0.000111720 |
| positive / negative / zero cases | 14 / 42 / 0 |
| candidate API total | 203.191s |
| parent API total | 172.065s |
| candidate calibration API | 161.746s |
| parent calibration API | 130.118s |

The mechanism was reachable and changed hard outputs, but the paired shard0 result regressed, so this concrete implementation is closed as `REJECTED` and does not replace v202. No official score or time is inferred.
