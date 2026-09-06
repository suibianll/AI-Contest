# proxy-v3 diagnosis

**Decision: `continue_next_shard`**

## Warnings

- no official-time prediction: run a fresh default panel without calibration cache

## Accuracy localization

- linear: delta_mean `+0.001688`, L1 `0.002400`, delta_tail `+0.000365`, +/-/0 `41/14/1`
  - role `q`: mean `+0.000327`, min `-0.002749`
  - layer `0`: mean `+0.000429`, min `-0.004653`
  - shape_bucket `hidden_to_hidden`: mean `+0.000529`, min `-0.002749`
  - role `fc_up`: mean `+0.000647`, min `-0.001356`

## Runtime localization

- `hif4_dynamic_quantize_activation`: 17.812s (100.0%), 56 calls
- `hif4_calibration_and_quantize_weight`: 0.000s (0.0%), 0 calls
- `hif4_calibration_attention`: 0.000s (0.0%), 0 calls
- `hif4_dynamic_quantize_q`: 0.000s (0.0%), 0 calls
- `hif4_dynamic_quantize_k`: 0.000s (0.0%), 0 calls
- `hif4_dynamic_quantize_v`: 0.000s (0.0%), 0 calls
- stages: calibration wall/API `0.000/0.000s`, scoring wall/API `22.490/17.812s`, cache load `1.640s`
- predicted official time: unavailable (requires fresh default panel)

## Next actions

- move work from online Activation quantization into calibration state
- inspect linear role=q first (mean delta=+0.000327, min delta=-0.002749)

> This tool never predicts an official score.
