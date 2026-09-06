# proxy-v3 diagnosis

**Decision: `continue_next_shard`**

## Warnings

- no official-time prediction: run a fresh default panel without calibration cache

## Accuracy localization

- linear: delta_mean `+0.001939`, L1 `0.003062`, delta_tail `+0.000951`, +/-/0 `40/16/0`
  - role_family `o`: mean `-0.002382`, min `-0.009491`
  - role `o`: mean `-0.002382`, min `-0.009491`
  - shape_bucket `hidden_to_hidden`: mean `-0.000973`, min `-0.009491`
  - layer `11`: mean `-0.000353`, min `-0.009491`

## Runtime localization

- `hif4_calibration_and_quantize_weight`: 46.067s (73.1%), 28 calls
- `hif4_dynamic_quantize_activation`: 16.911s (26.9%), 56 calls
- `hif4_calibration_attention`: 0.000s (0.0%), 0 calls
- `hif4_dynamic_quantize_q`: 0.000s (0.0%), 0 calls
- `hif4_dynamic_quantize_k`: 0.000s (0.0%), 0 calls
- `hif4_dynamic_quantize_v`: 0.000s (0.0%), 0 calls
- stages: calibration wall/API `47.292/46.067s`, scoring wall/API `21.277/16.911s`, cache load `0.000s`
- predicted official time: unavailable (requires fresh default panel)

## Next actions

- profile Weight calibration candidate loops, matrix factorizations, and repeated quantization
- inspect linear role_family=o first (mean delta=-0.002382, min delta=-0.009491)

> This tool never predicts an official score.
