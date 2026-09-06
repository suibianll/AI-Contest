# proxy-v3 diagnosis

**Decision: `continue_next_shard`**

## Warnings

- no official-time prediction: run a fresh default panel without calibration cache

## Accuracy localization

- linear: delta_mean `+0.003423`, L1 `0.004638`, delta_tail `+0.011872`, +/-/0 `38/16/2`
  - role_family `o`: mean `-0.001792`, min `-0.009374`
  - role `o`: mean `-0.001792`, min `-0.009374`
  - shape_bucket `hidden_to_hidden`: mean `-0.000486`, min `-0.009374`
  - role `v`: mean `-0.000238`, min `-0.004408`

## Runtime localization

- `hif4_calibration_and_quantize_weight`: 46.018s (73.0%), 28 calls
- `hif4_dynamic_quantize_activation`: 17.010s (27.0%), 56 calls
- `hif4_calibration_attention`: 0.000s (0.0%), 0 calls
- `hif4_dynamic_quantize_q`: 0.000s (0.0%), 0 calls
- `hif4_dynamic_quantize_k`: 0.000s (0.0%), 0 calls
- `hif4_dynamic_quantize_v`: 0.000s (0.0%), 0 calls
- stages: calibration wall/API `47.265/46.018s`, scoring wall/API `21.359/17.010s`, cache load `0.000s`
- predicted official time: unavailable (requires fresh default panel)

## Next actions

- profile Weight calibration candidate loops, matrix factorizations, and repeated quantization
- inspect linear role_family=o first (mean delta=-0.001792, min delta=-0.009374)

> This tool never predicts an official score.
