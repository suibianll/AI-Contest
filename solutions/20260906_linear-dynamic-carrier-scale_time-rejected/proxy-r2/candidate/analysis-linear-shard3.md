# proxy-v3 diagnosis

**Decision: `continue_next_shard`**

## Warnings

- no official-time prediction: run a fresh default panel without calibration cache

## Accuracy localization

- linear: delta_mean `+0.000314`, L1 `0.003133`, delta_tail `+0.000048`, +/-/0 `40/16/0`
  - role_family `o`: mean `-0.007815`, min `-0.023815`
  - role `o`: mean `-0.007815`, min `-0.023815`
  - shape_bucket `hidden_to_hidden`: mean `-0.004021`, min `-0.023815`
  - layer `21`: mean `-0.000649`, min `-0.023815`

## Runtime localization

- `hif4_calibration_and_quantize_weight`: 46.184s (73.1%), 28 calls
- `hif4_dynamic_quantize_activation`: 16.989s (26.9%), 56 calls
- `hif4_calibration_attention`: 0.000s (0.0%), 0 calls
- `hif4_dynamic_quantize_q`: 0.000s (0.0%), 0 calls
- `hif4_dynamic_quantize_k`: 0.000s (0.0%), 0 calls
- `hif4_dynamic_quantize_v`: 0.000s (0.0%), 0 calls
- stages: calibration wall/API `47.423/46.184s`, scoring wall/API `21.409/16.989s`, cache load `0.000s`
- predicted official time: unavailable (requires fresh default panel)

## Next actions

- profile Weight calibration candidate loops, matrix factorizations, and repeated quantization
- inspect linear role_family=o first (mean delta=-0.007815, min delta=-0.023815)

> This tool never predicts an official score.
