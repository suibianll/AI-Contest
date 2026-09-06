# proxy-v3 diagnosis

**Decision: `continue_next_shard`**

## Warnings

- no official-time prediction: run a fresh default panel without calibration cache

## Accuracy localization

- linear: delta_mean `+0.001710`, L1 `0.003236`, delta_tail `+0.002816`, +/-/0 `40/16/0`
  - role `v`: mean `-0.001344`, min `-0.004820`
  - layer `10`: mean `-0.000690`, min `-0.014468`
  - role_family `qkv`: mean `-0.000444`, min `-0.007753`
  - role `q`: mean `-0.000038`, min `-0.001705`

## Runtime localization

- `hif4_calibration_and_quantize_weight`: 46.093s (72.9%), 28 calls
- `hif4_dynamic_quantize_activation`: 17.151s (27.1%), 56 calls
- `hif4_calibration_attention`: 0.000s (0.0%), 0 calls
- `hif4_dynamic_quantize_q`: 0.000s (0.0%), 0 calls
- `hif4_dynamic_quantize_k`: 0.000s (0.0%), 0 calls
- `hif4_dynamic_quantize_v`: 0.000s (0.0%), 0 calls
- stages: calibration wall/API `47.317/46.093s`, scoring wall/API `21.521/17.151s`, cache load `0.000s`
- predicted official time: unavailable (requires fresh default panel)

## Next actions

- profile Weight calibration candidate loops, matrix factorizations, and repeated quantization
- inspect linear role=v first (mean delta=-0.001344, min delta=-0.004820)

> This tool never predicts an official score.
