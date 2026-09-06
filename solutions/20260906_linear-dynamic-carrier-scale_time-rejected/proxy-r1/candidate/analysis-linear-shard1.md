# proxy-v3 diagnosis

**Decision: `continue_next_shard`**

## Warnings

- linear: worst-20% tail regressed -0.001773
- no official-time prediction: run a fresh default panel without calibration cache

## Accuracy localization

- linear: delta_mean `+0.000050`, L1 `0.002443`, delta_tail `-0.001773`, +/-/0 `30/26/0`
  - role_family `o`: mean `-0.006030`, min `-0.012990`
  - role `o`: mean `-0.006030`, min `-0.012990`
  - shape_bucket `hidden_to_hidden`: mean `-0.002836`, min `-0.012990`
  - layer `1`: mean `-0.000518`, min `-0.008526`

## Runtime localization

- `hif4_calibration_and_quantize_weight`: 46.105s (72.9%), 28 calls
- `hif4_dynamic_quantize_activation`: 17.140s (27.1%), 56 calls
- `hif4_calibration_attention`: 0.000s (0.0%), 0 calls
- `hif4_dynamic_quantize_q`: 0.000s (0.0%), 0 calls
- `hif4_dynamic_quantize_k`: 0.000s (0.0%), 0 calls
- `hif4_dynamic_quantize_v`: 0.000s (0.0%), 0 calls
- stages: calibration wall/API `47.341/46.105s`, scoring wall/API `21.527/17.140s`, cache load `0.000s`
- predicted official time: unavailable (requires fresh default panel)

## Next actions

- profile Weight calibration candidate loops, matrix factorizations, and repeated quantization
- inspect linear role_family=o first (mean delta=-0.006030, min delta=-0.012990)

> This tool never predicts an official score.
