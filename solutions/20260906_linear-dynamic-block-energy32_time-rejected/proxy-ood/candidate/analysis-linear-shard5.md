# proxy-v3 diagnosis

**Decision: `continue_next_shard`**

## Warnings

- no official-time prediction: run a fresh default panel without calibration cache

## Accuracy localization

- linear: delta_mean `+0.003339`, L1 `0.003520`, delta_tail `+0.001234`, +/-/0 `50/6/0`
  - role `fc_up`: mean `+0.000589`, min `-0.001210`
  - role_family `fc`: mean `+0.000873`, min `-0.002887`
  - role `fc_gate`: mean `+0.001158`, min `-0.002887`
  - shape_bucket `hidden_to_wide`: mean `+0.001947`, min `-0.002887`

## Runtime localization

- `hif4_dynamic_quantize_activation`: 19.304s (100.0%), 56 calls
- `hif4_calibration_and_quantize_weight`: 0.000s (0.0%), 0 calls
- `hif4_calibration_attention`: 0.000s (0.0%), 0 calls
- `hif4_dynamic_quantize_q`: 0.000s (0.0%), 0 calls
- `hif4_dynamic_quantize_k`: 0.000s (0.0%), 0 calls
- `hif4_dynamic_quantize_v`: 0.000s (0.0%), 0 calls
- stages: calibration wall/API `0.000/0.000s`, scoring wall/API `23.939/19.304s`, cache load `1.428s`
- predicted official time: unavailable (requires fresh default panel)

## Next actions

- move work from online Activation quantization into calibration state
- inspect linear role=fc_up first (mean delta=+0.000589, min delta=-0.001210)

> This tool never predicts an official score.
