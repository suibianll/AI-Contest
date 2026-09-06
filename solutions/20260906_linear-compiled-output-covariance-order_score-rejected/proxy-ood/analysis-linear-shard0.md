# proxy-v3 diagnosis

**Decision: `reject`**

## Blockers

- linear: delta_mean=-0.000090 <= 0

## Warnings

- no official-time prediction: run a fresh default panel without calibration cache

## Accuracy localization

- linear: delta_mean `-0.000090`, L1 `0.001900`, delta_tail `+0.000619`, +/-/0 `22/26/8`
  - role `k`: mean `-0.002186`, min `-0.009625`
  - layer `18`: mean `-0.001271`, min `-0.009625`
  - role_family `qkv`: mean `-0.000836`, min `-0.009625`
  - role `fc_up`: mean `-0.000822`, min `-0.003932`

## Runtime localization

- `hif4_dynamic_quantize_activation`: 18.421s (100.0%), 56 calls
- `hif4_calibration_and_quantize_weight`: 0.000s (0.0%), 0 calls
- `hif4_calibration_attention`: 0.000s (0.0%), 0 calls
- `hif4_dynamic_quantize_q`: 0.000s (0.0%), 0 calls
- `hif4_dynamic_quantize_k`: 0.000s (0.0%), 0 calls
- `hif4_dynamic_quantize_v`: 0.000s (0.0%), 0 calls
- stages: calibration wall/API `0.000/0.000s`, scoring wall/API `22.742/18.421s`, cache load `1.407s`
- predicted official time: unavailable (requires fresh default panel)

## Next actions

- move work from online Activation quantization into calibration state
- inspect linear role=k first (mean delta=-0.002186, min delta=-0.009625)

> This tool never predicts an official score.
