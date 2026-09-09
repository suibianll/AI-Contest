# v209 — broadcast additive 4-element-group A@W

Status: `REJECTED` locally; official status `unregistered/NA`.

- Parent: retained v202 complete root, SHA256 `56dc805d6e5a3aef896db8021045740292735725d688b48e3d4393e55efcb2bd`
- Candidate SHA256: `1b82667b8813a1ad6fefee55f2d3843d63e78d859eb8b535659c06ba3308c36b`
- Mechanism: one additive value per deployed natural 4-element input group, broadcast across output rows,
  fitted from the actual output residual; `gptq_block_order` remains activation-only.
- Fit: all calibration pairs and all rows supplied to the API, one fixed diagonal residual quadratic solve,
  one legal five-field re-encode, and a hard-output gate against the parent.

## Local result

`eval-v3`, Qwen3.5-4B proxy-v2, Linear-only, all six target-side shards (336 cases):

| candidate mean | parent mean | delta | changed cases |
|---:|---:|---:|---:|
| 0.5292658476834804 | 0.5292658476834804 | 0 | 0 |

All outputs were finite and all cases were covered. The candidate does not replace v202.
The diagnostic API total was `1298.568s`; local timing is not an official-time claim.

The controlled fit reached `attempted=1`, `accepted=0`; hard-output losses were `1087.6863708` for the
parent and `1837.1724854` for the re-encoded candidate. No official result is inferred from the local run.
