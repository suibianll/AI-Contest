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

- `hif4_dynamic_quantize_activation`: 17.170s (100.0%), 56 calls
- `hif4_calibration_and_quantize_weight`: 0.000s (0.0%), 0 calls
- `hif4_calibration_attention`: 0.000s (0.0%), 0 calls
- `hif4_dynamic_quantize_q`: 0.000s (0.0%), 0 calls
- `hif4_dynamic_quantize_k`: 0.000s (0.0%), 0 calls
- `hif4_dynamic_quantize_v`: 0.000s (0.0%), 0 calls
- stages: calibration wall/API `0.000/0.000s`, scoring wall/API `21.541/17.170s`, cache load `1.562s`
- predicted official time: unavailable (requires fresh default panel)

## Next actions

- move work from online Activation quantization into calibration state
- inspect linear layer=0 first (mean delta=-0.000294, min delta=-0.003258)

> This tool never predicts an official score.
