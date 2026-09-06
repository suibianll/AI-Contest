# proxy-v3 diagnosis

**Decision: `continue_next_shard`**

## Warnings

- linear: worst-20% tail regressed -0.000065
- no official-time prediction: run a fresh default panel without calibration cache

## Accuracy localization

- linear: delta_mean `+0.001063`, L1 `0.003510`, delta_tail `-0.000065`, +/-/0 `29/27/0`
  - role_family `o`: mean `-0.002598`, min `-0.007439`
  - role `o`: mean `-0.002598`, min `-0.007439`
  - shape_bucket `hidden_to_hidden`: mean `-0.002252`, min `-0.011206`
  - role `q`: mean `-0.001906`, min `-0.011206`

## Runtime localization

- `hif4_dynamic_quantize_activation`: 18.530s (100.0%), 56 calls
- `hif4_calibration_and_quantize_weight`: 0.000s (0.0%), 0 calls
- `hif4_calibration_attention`: 0.000s (0.0%), 0 calls
- `hif4_dynamic_quantize_q`: 0.000s (0.0%), 0 calls
- `hif4_dynamic_quantize_k`: 0.000s (0.0%), 0 calls
- `hif4_dynamic_quantize_v`: 0.000s (0.0%), 0 calls
- stages: calibration wall/API `0.000/0.000s`, scoring wall/API `23.173/18.530s`, cache load `1.527s`
- predicted official time: unavailable (requires fresh default panel)

## Next actions

- move work from online Activation quantization into calibration state
- inspect linear role_family=o first (mean delta=-0.002598, min delta=-0.007439)

> This tool never predicts an official score.
