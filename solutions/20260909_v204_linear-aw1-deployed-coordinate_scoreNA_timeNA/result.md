# v204 — deployment-coordinate 64-block A@W

Status: `REJECTED` locally; official status `unregistered/NA`.

- Parent: retained v202 complete root, SHA256 `56dc805d6e5a3aef896db8021045740292735725d688b48e3d4393e55efcb2bd`
- Candidate SHA256: `b4a5b1551329166f8e5f39ec042ade19d0714c413cc42f45a8207749476ea679`
- Mechanism: one scalar gain per 64-channel block in the final natural deployment coordinate.
  `gptq_block_order` is used only by the activation execution path and is not reapplied to weight blocks.
- Fit: all calibration pairs and all rows supplied to the API; target is the actual output `XW^T` with
  the parent hard `Q(X)` held fixed. The re-encoded five-field candidate is accepted only after a hard-output comparison.

## Local result

`eval-v3`, Qwen3.5-4B proxy-v2, Linear-only, all six target-side shards (336 cases):

| candidate mean | parent mean | delta | changed cases |
|---:|---:|---:|---:|
| 0.5292658476834804 | 0.5292658476834804 | 0 | 0 |

All outputs were finite and all cases were covered. The candidate therefore does not replace v202.
The diagnostic API total was `1279.124s` versus the v202 reference `1021.601s`; this is local diagnostic data,
not an official-time claim.

The controlled fit reached `attempted=1`, `accepted=0`; its hard-output losses were `14.9340868` for the
parent and `23.7784224` for the re-encoded candidate. The implementation records these fields per calibration
state. No official result is inferred from the local run.
