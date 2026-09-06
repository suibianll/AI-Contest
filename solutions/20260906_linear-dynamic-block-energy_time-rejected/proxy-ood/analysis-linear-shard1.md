# proxy-v3 diagnosis

**Decision: `continue_next_shard`**

## Warnings

- no official-time prediction: run a fresh default panel without calibration cache

## Accuracy localization

- linear: delta_mean `+0.002877`, L1 `0.003114`, delta_tail `+0.001195`, +/-/0 `50/6/0`
  - role `fc_up`: mean `+0.000824`, min `+0.000198`
  - role_family `fc`: mean `+0.000908`, min `+0.000023`
  - role `q`: mean `+0.000984`, min `-0.000972`
  - role `fc_gate`: mean `+0.000991`, min `+0.000023`

## Runtime localization

- `hif4_dynamic_quantize_activation`: 18.594s (100.0%), 56 calls
- `hif4_calibration_and_quantize_weight`: 0.000s (0.0%), 0 calls
- `hif4_calibration_attention`: 0.000s (0.0%), 0 calls
- `hif4_dynamic_quantize_q`: 0.000s (0.0%), 0 calls
- `hif4_dynamic_quantize_k`: 0.000s (0.0%), 0 calls
- `hif4_dynamic_quantize_v`: 0.000s (0.0%), 0 calls
- stages: calibration wall/API `0.000/0.000s`, scoring wall/API `23.178/18.594s`, cache load `1.455s`
- predicted official time: unavailable (requires fresh default panel)

## Next actions

- move work from online Activation quantization into calibration state
- inspect linear role=fc_up first (mean delta=+0.000824, min delta=+0.000198)

> This tool never predicts an official score.
