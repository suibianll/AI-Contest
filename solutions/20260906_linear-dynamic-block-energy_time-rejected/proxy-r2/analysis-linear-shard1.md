# proxy-v3 diagnosis

**Decision: `continue_next_shard`**

## Warnings

- no official-time prediction: run a fresh default panel without calibration cache

## Accuracy localization

- linear: delta_mean `+0.001893`, L1 `0.002144`, delta_tail `+0.000987`, +/-/0 `46/10/0`
  - role `fc_gate`: mean `+0.000570`, min `-0.000476`
  - role `v`: mean `+0.000573`, min `-0.002229`
  - shape_bucket `hidden_to_wide`: mean `+0.000764`, min `-0.002229`
  - role_family `fc`: mean `+0.000804`, min `-0.000476`

## Runtime localization

- `hif4_dynamic_quantize_activation`: 17.070s (100.0%), 56 calls
- `hif4_calibration_and_quantize_weight`: 0.000s (0.0%), 0 calls
- `hif4_calibration_attention`: 0.000s (0.0%), 0 calls
- `hif4_dynamic_quantize_q`: 0.000s (0.0%), 0 calls
- `hif4_dynamic_quantize_k`: 0.000s (0.0%), 0 calls
- `hif4_dynamic_quantize_v`: 0.000s (0.0%), 0 calls
- stages: calibration wall/API `0.000/0.000s`, scoring wall/API `21.462/17.070s`, cache load `1.496s`
- predicted official time: unavailable (requires fresh default panel)

## Next actions

- move work from online Activation quantization into calibration state
- inspect linear role=fc_gate first (mean delta=+0.000570, min delta=-0.000476)

> This tool never predicts an official score.
