# proxy-v3 diagnosis

**Decision: `continue_next_shard`**

## Warnings

- no official-time prediction: run a fresh default panel without calibration cache

## Accuracy localization

- linear: delta_mean `+0.001397`, L1 `0.003246`, delta_tail `+0.000058`, +/-/0 `32/24/0`
  - role_family `o`: mean `-0.001588`, min `-0.005583`
  - role `o`: mean `-0.001588`, min `-0.005583`
  - layer `13`: mean `-0.001211`, min `-0.013586`
  - role `k`: mean `-0.000259`, min `-0.013586`

## Runtime localization

- `hif4_dynamic_quantize_activation`: 18.577s (100.0%), 56 calls
- `hif4_calibration_and_quantize_weight`: 0.000s (0.0%), 0 calls
- `hif4_calibration_attention`: 0.000s (0.0%), 0 calls
- `hif4_dynamic_quantize_q`: 0.000s (0.0%), 0 calls
- `hif4_dynamic_quantize_k`: 0.000s (0.0%), 0 calls
- `hif4_dynamic_quantize_v`: 0.000s (0.0%), 0 calls
- stages: calibration wall/API `0.000/0.000s`, scoring wall/API `23.149/18.577s`, cache load `1.548s`
- predicted official time: unavailable (requires fresh default panel)

## Next actions

- move work from online Activation quantization into calibration state
- inspect linear role_family=o first (mean delta=-0.001588, min delta=-0.005583)

> This tool never predicts an official score.
