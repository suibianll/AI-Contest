# v206 — output-group × 64-block A@W

Status: `REJECTED` locally; official status `unregistered/NA`.

- Parent: retained v202 complete root, SHA256 `56dc805d6e5a3aef896db8021045740292735725d688b48e3d4393e55efcb2bd`
- Candidate SHA256: `7b367bfc5315ee2b495bd8009ca153c2e7257dd5ee28ee70d4be068806e9f5da`
- Mechanism: one scalar per deployed input 64-block shared within each fixed output row group of 64,
  fitted in the final natural deployment coordinate; `gptq_block_order` remains activation-only.
- Fit: all calibration pairs and all rows supplied to the API, one batched closed-form solve, one legal
  five-field re-encode, and a hard-output gate against the parent.

## Local result

`eval-v3`, Qwen3.5-4B proxy-v2, Linear-only, all six target-side shards (336 cases):

| candidate mean | parent mean | delta | changed cases |
|---:|---:|---:|---:|
| 0.5292658476834804 | 0.5292658476834804 | 0 | 0 |

All outputs were finite and all cases were covered. The candidate does not replace v202.
The diagnostic API total was `1291.923s`; local timing is not an official-time claim.

The controlled fit reached `attempted=1`, `accepted=0`; hard-output losses were `968.4084473` for the
parent and `1389.3823853` for the re-encoded candidate. No official result is inferred from the local run.
