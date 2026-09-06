# proxy-v3 diagnosis

**Decision: `continue_next_shard`**

## Warnings

- no official-time prediction: run a fresh default panel without calibration cache

## Accuracy localization

- linear: delta_mean `+0.000813`, L1 `0.001542`, delta_tail `+0.000976`, +/-/0 `38/10/8`
  - role_family `o`: mean `-0.000634`, min `-0.007256`
  - role `o`: mean `-0.000634`, min `-0.007256`
  - layer `15`: mean `-0.000360`, min `-0.007256`
  - shape_bucket `hidden_to_hidden`: mean `-0.000145`, min `-0.007256`

## Runtime localization

- `hif4_calibration_and_quantize_weight`: 44.971s (72.9%), 28 calls
- `hif4_dynamic_quantize_activation`: 16.739s (27.1%), 56 calls
- `hif4_calibration_attention`: 0.000s (0.0%), 0 calls
- `hif4_dynamic_quantize_q`: 0.000s (0.0%), 0 calls
- `hif4_dynamic_quantize_k`: 0.000s (0.0%), 0 calls
- `hif4_dynamic_quantize_v`: 0.000s (0.0%), 0 calls
- stages: calibration wall/API `46.149/44.971s`, scoring wall/API `20.751/16.739s`, cache load `0.000s`
- predicted official time: unavailable (requires fresh default panel)

## Next actions

- profile Weight calibration candidate loops, matrix factorizations, and repeated quantization
- inspect linear role_family=o first (mean delta=-0.000634, min delta=-0.007256)

> This tool never predicts an official score.
