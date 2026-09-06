# proxy-v3 diagnosis

**Decision: `continue_next_shard`**

## Warnings

- no official-time prediction: run a fresh default panel without calibration cache

## Accuracy localization

- linear: delta_mean `+0.000732`, L1 `0.002365`, delta_tail `+0.000896`, +/-/0 `28/20/8`
  - role `k`: mean `-0.001653`, min `-0.011579`
  - role `v`: mean `-0.001131`, min `-0.006579`
  - role_family `qkv`: mean `-0.001097`, min `-0.011579`
  - length `512`: mean `-0.000688`, min `-0.003281`

## Runtime localization

- `hif4_dynamic_quantize_activation`: 18.160s (100.0%), 56 calls
- `hif4_calibration_and_quantize_weight`: 0.000s (0.0%), 0 calls
- `hif4_calibration_attention`: 0.000s (0.0%), 0 calls
- `hif4_dynamic_quantize_q`: 0.000s (0.0%), 0 calls
- `hif4_dynamic_quantize_k`: 0.000s (0.0%), 0 calls
- `hif4_dynamic_quantize_v`: 0.000s (0.0%), 0 calls
- stages: calibration wall/API `0.000/0.000s`, scoring wall/API `22.415/18.160s`, cache load `1.370s`
- predicted official time: unavailable (requires fresh default panel)

## Next actions

- move work from online Activation quantization into calibration state
- inspect linear role=k first (mean delta=-0.001653, min delta=-0.011579)

> This tool never predicts an official score.
