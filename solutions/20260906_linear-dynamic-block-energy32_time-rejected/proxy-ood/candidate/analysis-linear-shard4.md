# proxy-v3 diagnosis

**Decision: `continue_next_shard`**

## Warnings

- no official-time prediction: run a fresh default panel without calibration cache

## Accuracy localization

- linear: delta_mean `+0.002813`, L1 `0.003285`, delta_tail `+0.002567`, +/-/0 `48/8/0`
  - role `k`: mean `+0.000194`, min `-0.005933`
  - role `fc_gate`: mean `+0.000581`, min `-0.003369`
  - shape_bucket `hidden_to_wide`: mean `+0.000793`, min `-0.005933`
  - role_family `fc`: mean `+0.000832`, min `-0.003369`

## Runtime localization

- `hif4_dynamic_quantize_activation`: 18.694s (100.0%), 56 calls
- `hif4_calibration_and_quantize_weight`: 0.000s (0.0%), 0 calls
- `hif4_calibration_attention`: 0.000s (0.0%), 0 calls
- `hif4_dynamic_quantize_q`: 0.000s (0.0%), 0 calls
- `hif4_dynamic_quantize_k`: 0.000s (0.0%), 0 calls
- `hif4_dynamic_quantize_v`: 0.000s (0.0%), 0 calls
- stages: calibration wall/API `0.000/0.000s`, scoring wall/API `23.094/18.694s`, cache load `1.469s`
- predicted official time: unavailable (requires fresh default panel)

## Next actions

- move work from online Activation quantization into calibration state
- inspect linear role=k first (mean delta=+0.000194, min delta=-0.005933)

> This tool never predicts an official score.
