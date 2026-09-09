# v208 — output-row 8-group × 64-block A@W

Status: `REJECTED` locally; official status `unregistered/NA`.

- Parent: retained v202 complete root, SHA256 `56dc805d6e5a3aef896db8021045740292735725d688b48e3d4393e55efcb2bd`
- Candidate SHA256: `c93d48b09899e4c136149e570257f45f8930fd5d871fa4a4f3d27377f50af5dc`
- Mechanism: one scalar per deployed input 64-block shared within each fixed output row group of 8,
  fitted in the final natural deployment coordinate; `gptq_block_order` remains activation-only.
- Fit: all calibration pairs and all rows supplied to the API, one batched closed-form solve, one legal
  five-field re-encode, and a hard-output gate against the parent.

## Local result

`eval-v3`, Qwen3.5-4B proxy-v2, Linear-only, all six target-side shards (336 cases):

| candidate mean | parent mean | delta | changed cases |
|---:|---:|---:|---:|
| 0.5292658476834804 | 0.5292658476834804 | 0 | 0 |

All outputs were finite and all cases were covered. The candidate does not replace v202.
The diagnostic API total was `1302.458s`; local timing is not an official-time claim.

The controlled fit reached `attempted=1`, `accepted=0`; hard-output losses were `1002.1207886` for the
parent and `1630.4505005` for the re-encoded candidate. No official result is inferred from the local run.
