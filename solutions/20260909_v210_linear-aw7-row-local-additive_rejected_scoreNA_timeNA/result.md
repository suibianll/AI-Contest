# v210 — row-local additive 4-element-group A@W

Status: `REJECTED` locally; official status `unregistered/NA`.

- Parent: retained v202 complete root, SHA256 `56dc805d6e5a3aef896db8021045740292735725d688b48e3d4393e55efcb2bd`
- Candidate SHA256: `d99aec6cfceac9e49119bf9b4dd8241d201f089add36be9b47bc4d9039149e2a`
- Mechanism: one additive value per deployed natural 4-element input group for each output row,
  fitted from the actual output residual; `gptq_block_order` remains activation-only.
- Fit: all calibration pairs and all rows supplied to the API, one fixed diagonal residual quadratic solve,
  one legal five-field re-encode, and a hard-output gate against the parent.

## Local result

`eval-v3`, Qwen3.5-4B proxy-v2, Linear-only, all six target-side shards (336 cases):

| candidate mean | parent mean | delta | changed cases |
|---:|---:|---:|---:|
| 0.5292658476834804 | 0.5292658476834804 | 0 | 0 |

All outputs were finite and all cases were covered. The candidate does not replace v202.
The diagnostic API total was `1454.486s`; local timing is not an official-time claim.

The controlled fit reached `attempted=1`, `accepted=0`; the hard-output candidate was not accepted.
No official result is inferred from the local run.
