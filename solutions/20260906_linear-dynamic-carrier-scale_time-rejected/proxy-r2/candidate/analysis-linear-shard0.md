# proxy-v3 diagnosis

**Decision: `continue_next_shard`**

## Warnings

- linear: worst-20% tail regressed -0.000196
- no official-time prediction: run a fresh default panel without calibration cache

## Accuracy localization

- linear: delta_mean `+0.001170`, L1 `0.002063`, delta_tail `-0.000196`, +/-/0 `38/18/0`
  - role_family `o`: mean `-0.001036`, min `-0.005179`
  - role `o`: mean `-0.001036`, min `-0.005179`
  - shape_bucket `hidden_to_hidden`: mean `-0.000495`, min `-0.005179`
  - role `q`: mean `+0.000046`, min `-0.002509`

## Runtime localization

- `hif4_dynamic_quantize_activation`: 17.094s (100.0%), 56 calls
- `hif4_calibration_and_quantize_weight`: 0.000s (0.0%), 0 calls
- `hif4_calibration_attention`: 0.000s (0.0%), 0 calls
- `hif4_dynamic_quantize_q`: 0.000s (0.0%), 0 calls
- `hif4_dynamic_quantize_k`: 0.000s (0.0%), 0 calls
- `hif4_dynamic_quantize_v`: 0.000s (0.0%), 0 calls
- stages: calibration wall/API `0.000/0.000s`, scoring wall/API `21.394/17.094s`, cache load `1.679s`
- predicted official time: unavailable (requires fresh default panel)

## Next actions

- move work from online Activation quantization into calibration state
- inspect linear role_family=o first (mean delta=-0.001036, min delta=-0.005179)

> This tool never predicts an official score.
