# proxy-v3 diagnosis

**Decision: `continue_next_shard`**

## Warnings

- no official-time prediction: run a fresh default panel without calibration cache

## Accuracy localization

- linear: delta_mean `+0.002713`, L1 `0.003705`, delta_tail `+0.003964`, +/-/0 `45/11/0`
  - role `k`: mean `-0.001662`, min `-0.012119`
  - role_family `qkv`: mean `+0.000272`, min `-0.012119`
  - role `fc_gate`: mean `+0.000943`, min `-0.000495`
  - role `q`: mean `+0.001060`, min `-0.000885`

## Runtime localization

- `hif4_dynamic_quantize_activation`: 18.483s (100.0%), 56 calls
- `hif4_calibration_and_quantize_weight`: 0.000s (0.0%), 0 calls
- `hif4_calibration_attention`: 0.000s (0.0%), 0 calls
- `hif4_dynamic_quantize_q`: 0.000s (0.0%), 0 calls
- `hif4_dynamic_quantize_k`: 0.000s (0.0%), 0 calls
- `hif4_dynamic_quantize_v`: 0.000s (0.0%), 0 calls
- stages: calibration wall/API `0.000/0.000s`, scoring wall/API `22.847/18.483s`, cache load `1.324s`
- predicted official time: unavailable (requires fresh default panel)

## Next actions

- move work from online Activation quantization into calibration state
- inspect linear role=k first (mean delta=-0.001662, min delta=-0.012119)

> This tool never predicts an official score.
