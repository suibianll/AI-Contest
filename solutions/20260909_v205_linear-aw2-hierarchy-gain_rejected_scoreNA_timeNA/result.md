# v205 — hierarchy-aligned 8-element-group A@W

Status: `REJECTED` locally; official status `unregistered/NA`.

- Parent: retained v202 complete root, SHA256 `56dc805d6e5a3aef896db8021045740292735725d688b48e3d4393e55efcb2bd`
- Candidate SHA256: `741cc0f8ccb5d84416a63064027728c07bb51b523c9b701336f81259cd75d874`
- Mechanism: one scalar gain per 8-element group in the final natural deployment coordinate, matching
  the first HiF4 hierarchy level; `gptq_block_order` remains activation-only.
- Fit: all calibration pairs and all rows supplied to the API, followed by one legal five-field re-encode
  and a hard-output gate against the parent.

## Local result

`eval-v3`, Qwen3.5-4B proxy-v2, Linear-only, all six target-side shards (336 cases):

| candidate mean | parent mean | delta | changed cases |
|---:|---:|---:|---:|
| 0.5292658476834804 | 0.5292658476834804 | 0 | 0 |

All outputs were finite and all cases were covered. The candidate does not replace v202.
The diagnostic API total was `1296.177s`; local timing is not an official-time claim.

The controlled fit reached `attempted=1`, `accepted=0`; hard-output losses were `15.5380673` for the
parent and `72.7920227` for the re-encoded candidate. No official result is inferred from the local run.
