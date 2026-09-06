# proxy-v3 diagnosis

**Decision: `continue_next_shard`**

## Warnings

- no official-time prediction: run a fresh default panel without calibration cache

## Accuracy localization

- linear: delta_mean `+0.002985`, L1 `0.003511`, delta_tail `+0.001087`, +/-/0 `46/10/0`
  - role `q`: mean `-0.000132`, min `-0.002279`
  - role `fc_up`: mean `+0.000652`, min `-0.000116`
  - layer `0`: mean `+0.001202`, min `-0.002279`
  - role_family `qkv`: mean `+0.001379`, min `-0.002935`

## Runtime localization

- `hif4_dynamic_quantize_activation`: 18.521s (100.0%), 56 calls
- `hif4_calibration_and_quantize_weight`: 0.000s (0.0%), 0 calls
- `hif4_calibration_attention`: 0.000s (0.0%), 0 calls
- `hif4_dynamic_quantize_q`: 0.000s (0.0%), 0 calls
- `hif4_dynamic_quantize_k`: 0.000s (0.0%), 0 calls
- `hif4_dynamic_quantize_v`: 0.000s (0.0%), 0 calls
- stages: calibration wall/API `0.000/0.000s`, scoring wall/API `23.224/18.521s`, cache load `1.514s`
- predicted official time: unavailable (requires fresh default panel)

## Next actions

- move work from online Activation quantization into calibration state
- inspect linear role=q first (mean delta=-0.000132, min delta=-0.002279)

> This tool never predicts an official score.
