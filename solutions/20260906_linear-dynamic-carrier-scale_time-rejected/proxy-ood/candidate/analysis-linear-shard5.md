# proxy-v3 diagnosis

**Decision: `continue_next_shard`**

## Warnings

- no official-time prediction: run a fresh default panel without calibration cache

## Accuracy localization

- linear: delta_mean `+0.001381`, L1 `0.004440`, delta_tail `+0.000412`, +/-/0 `39/17/0`
  - role_family `o`: mean `-0.005920`, min `-0.012322`
  - role `o`: mean `-0.005920`, min `-0.012322`
  - shape_bucket `hidden_to_hidden`: mean `-0.002917`, min `-0.012322`
  - role `k`: mean `-0.001438`, min `-0.019342`

## Runtime localization

- `hif4_dynamic_quantize_activation`: 18.475s (100.0%), 56 calls
- `hif4_calibration_and_quantize_weight`: 0.000s (0.0%), 0 calls
- `hif4_calibration_attention`: 0.000s (0.0%), 0 calls
- `hif4_dynamic_quantize_q`: 0.000s (0.0%), 0 calls
- `hif4_dynamic_quantize_k`: 0.000s (0.0%), 0 calls
- `hif4_dynamic_quantize_v`: 0.000s (0.0%), 0 calls
- stages: calibration wall/API `0.000/0.000s`, scoring wall/API `23.127/18.475s`, cache load `1.517s`
- predicted official time: unavailable (requires fresh default panel)

## Next actions

- move work from online Activation quantization into calibration state
- inspect linear role_family=o first (mean delta=-0.005920, min delta=-0.012322)

> This tool never predicts an official score.
