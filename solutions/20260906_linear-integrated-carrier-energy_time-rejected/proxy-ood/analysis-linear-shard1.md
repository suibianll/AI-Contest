# proxy-v3 diagnosis

**Decision: `continue_next_shard`**

## Warnings

- no official-time prediction: run a fresh default panel without calibration cache

## Accuracy localization

- linear: delta_mean `+0.000217`, L1 `0.001620`, delta_tail `+0.000259`, +/-/0 `31/23/2`
  - role `v`: mean `-0.001111`, min `-0.006294`
  - role_family `proj`: mean `-0.000763`, min `-0.002721`
  - role `proj`: mean `-0.000763`, min `-0.002721`
  - shape_bucket `wide_to_hidden`: mean `-0.000763`, min `-0.002721`

## Runtime localization

- `hif4_dynamic_quantize_activation`: 18.463s (100.0%), 56 calls
- `hif4_calibration_and_quantize_weight`: 0.000s (0.0%), 0 calls
- `hif4_calibration_attention`: 0.000s (0.0%), 0 calls
- `hif4_dynamic_quantize_q`: 0.000s (0.0%), 0 calls
- `hif4_dynamic_quantize_k`: 0.000s (0.0%), 0 calls
- `hif4_dynamic_quantize_v`: 0.000s (0.0%), 0 calls
- stages: calibration wall/API `0.000/0.000s`, scoring wall/API `23.250/18.463s`, cache load `1.432s`
- predicted official time: unavailable (requires fresh default panel)

## Next actions

- move work from online Activation quantization into calibration state
- inspect linear role=v first (mean delta=-0.001111, min delta=-0.006294)

> This tool never predicts an official score.
