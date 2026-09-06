# proxy-v3 diagnosis

**Decision: `continue_next_shard`**

## Warnings

- no official-time prediction: run a fresh default panel without calibration cache

## Accuracy localization

- linear: delta_mean `+0.004866`, L1 `0.005562`, delta_tail `+0.007217`, +/-/0 `42/14/0`
  - role `fc_up`: mean `+0.000898`, min `-0.000548`
  - layer `14`: mean `+0.001454`, min `-0.002919`
  - layer `8`: mean `+0.001540`, min `-0.004518`
  - role `q`: mean `+0.001786`, min `-0.002919`

## Runtime localization

- `hif4_dynamic_quantize_activation`: 18.842s (100.0%), 56 calls
- `hif4_calibration_and_quantize_weight`: 0.000s (0.0%), 0 calls
- `hif4_calibration_attention`: 0.000s (0.0%), 0 calls
- `hif4_dynamic_quantize_q`: 0.000s (0.0%), 0 calls
- `hif4_dynamic_quantize_k`: 0.000s (0.0%), 0 calls
- `hif4_dynamic_quantize_v`: 0.000s (0.0%), 0 calls
- stages: calibration wall/API `0.000/0.000s`, scoring wall/API `23.578/18.842s`, cache load `1.528s`
- predicted official time: unavailable (requires fresh default panel)

## Next actions

- move work from online Activation quantization into calibration state
- inspect linear role=fc_up first (mean delta=+0.000898, min delta=-0.000548)

> This tool never predicts an official score.
