# proxy-v3 diagnosis

**Decision: `reject`**

## Blockers

- linear: delta_mean=-0.000431 <= 0

## Warnings

- no official-time prediction: run a fresh default panel without calibration cache

## Accuracy localization

- linear: delta_mean `-0.000431`, L1 `0.004385`, delta_tail `+0.002526`, +/-/0 `33/23/0`
  - role_family `o`: mean `-0.009851`, min `-0.038428`
  - role `o`: mean `-0.009851`, min `-0.038428`
  - shape_bucket `hidden_to_hidden`: mean `-0.005216`, min `-0.038428`
  - length `128`: mean `-0.004790`, min `-0.038428`

## Runtime localization

- `hif4_dynamic_quantize_activation`: 19.531s (100.0%), 56 calls
- `hif4_calibration_and_quantize_weight`: 0.000s (0.0%), 0 calls
- `hif4_calibration_attention`: 0.000s (0.0%), 0 calls
- `hif4_dynamic_quantize_q`: 0.000s (0.0%), 0 calls
- `hif4_dynamic_quantize_k`: 0.000s (0.0%), 0 calls
- `hif4_dynamic_quantize_v`: 0.000s (0.0%), 0 calls
- stages: calibration wall/API `0.000/0.000s`, scoring wall/API `24.449/19.531s`, cache load `1.564s`
- predicted official time: unavailable (requires fresh default panel)

## Next actions

- move work from online Activation quantization into calibration state
- inspect linear role_family=o first (mean delta=-0.009851, min delta=-0.038428)

> This tool never predicts an official score.
