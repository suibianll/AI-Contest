# proxy-v3 diagnosis

**Decision: `continue_next_shard`**

## Warnings

- no official-time prediction: run a fresh default panel without calibration cache

## Accuracy localization

- linear: delta_mean `+0.003253`, L1 `0.003917`, delta_tail `+0.004336`, +/-/0 `46/10/0`
  - role `q`: mean `+0.000527`, min `-0.001063`
  - role `k`: mean `+0.000947`, min `-0.001915`
  - role_family `qkv`: mean `+0.000972`, min `-0.002323`
  - role `fc_gate`: mean `+0.001058`, min `-0.003907`

## Runtime localization

- `hif4_dynamic_quantize_activation`: 18.645s (100.0%), 56 calls
- `hif4_calibration_and_quantize_weight`: 0.000s (0.0%), 0 calls
- `hif4_calibration_attention`: 0.000s (0.0%), 0 calls
- `hif4_dynamic_quantize_q`: 0.000s (0.0%), 0 calls
- `hif4_dynamic_quantize_k`: 0.000s (0.0%), 0 calls
- `hif4_dynamic_quantize_v`: 0.000s (0.0%), 0 calls
- stages: calibration wall/API `0.000/0.000s`, scoring wall/API `23.057/18.645s`, cache load `1.480s`
- predicted official time: unavailable (requires fresh default panel)

## Next actions

- move work from online Activation quantization into calibration state
- inspect linear role=q first (mean delta=+0.000527, min delta=-0.001063)

> This tool never predicts an official score.
