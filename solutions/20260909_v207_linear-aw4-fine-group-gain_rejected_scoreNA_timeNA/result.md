# v207 — fine-grained 4-element-group A@W

Status: `REJECTED` locally; official status `unregistered/NA`.

- Parent: retained v202 complete root, SHA256 `56dc805d6e5a3aef896db8021045740292735725d688b48e3d4393e55efcb2bd`
- Candidate SHA256: `ac0d180de6698f330d4090cc6cdedfdb841dcf9c046fd1c641194fb0e0365ce7`
- Mechanism: one scalar per deployed 4-element HiF4 group, fitted in the final natural deployment
  coordinate; `gptq_block_order` remains activation-only.
- Fit: all calibration pairs and all rows supplied to the API, one fixed closed-form A@W fit, one legal
  five-field re-encode, and a hard-output gate against the parent.

## Local result

`eval-v3`, Qwen3.5-4B proxy-v2, Linear-only, all six target-side shards (336 cases):

| candidate mean | parent mean | delta | changed cases |
|---:|---:|---:|---:|
| 0.5292658476834804 | 0.5292658476834804 | 0 | 0 |

All outputs were finite and all cases were covered. The candidate does not replace v202.
The diagnostic API total was `1331.549s`; local timing is not an official-time claim.

The controlled fit reached `attempted=1`, `accepted=0`; hard-output losses were `15.5380673` for the
parent and `54.0371056` for the re-encoded candidate. No official result is inferred from the local run.
