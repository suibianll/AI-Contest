# proxy-v3 diagnosis

**Decision: `continue_next_shard`**

## Warnings

- no official-time prediction: run a fresh default panel without calibration cache

## Accuracy localization

- linear: delta_mean `+0.001252`, L1 `0.001840`, delta_tail `+0.000705`, +/-/0 `43/13/0`
  - layer `0`: mean `-0.000294`, min `-0.003258`
  - role `q`: mean `+0.000084`, min `-0.002305`
  - shape_bucket `hidden_to_hidden`: mean `+0.000261`, min `-0.002305`
  - role_family `o`: mean `+0.000439`, min `-0.001581`

## Runtime localization

- `hif4_calibration_and_quantize_weight`: 46.365s (73.3%), 28 calls
- `hif4_dynamic_quantize_activation`: 16.873s (26.7%), 56 calls
- `hif4_calibration_attention`: 0.000s (0.0%), 0 calls
- `hif4_dynamic_quantize_q`: 0.000s (0.0%), 0 calls
- `hif4_dynamic_quantize_k`: 0.000s (0.0%), 0 calls
- `hif4_dynamic_quantize_v`: 0.000s (0.0%), 0 calls
- stages: calibration wall/API `47.606/46.365s`, scoring wall/API `21.188/16.873s`, cache load `0.000s`
- predicted official time: unavailable (requires fresh default panel)

## Next actions

- profile Weight calibration candidate loops, matrix factorizations, and repeated quantization
- inspect linear layer=0 first (mean delta=-0.000294, min delta=-0.003258)

> This tool never predicts an official score.
