# proxy-v3 diagnosis

**Decision: `continue_next_shard`**

## Warnings

- no official-time prediction: run a fresh default panel without calibration cache

## Accuracy localization

- linear: delta_mean `+0.002511`, L1 `0.002767`, delta_tail `+0.000878`, +/-/0 `49/7/0`
  - length `1024`: mean `+0.000778`, min `-0.000572`
  - role `fc_up`: mean `+0.000818`, min `-0.000065`
  - role_family `fc`: mean `+0.000853`, min `-0.000065`
  - role `fc_gate`: mean `+0.000888`, min `+0.000023`

## Runtime localization

- `hif4_dynamic_quantize_activation`: 18.464s (100.0%), 56 calls
- `hif4_calibration_and_quantize_weight`: 0.000s (0.0%), 0 calls
- `hif4_calibration_attention`: 0.000s (0.0%), 0 calls
- `hif4_dynamic_quantize_q`: 0.000s (0.0%), 0 calls
- `hif4_dynamic_quantize_k`: 0.000s (0.0%), 0 calls
- `hif4_dynamic_quantize_v`: 0.000s (0.0%), 0 calls
- stages: calibration wall/API `0.000/0.000s`, scoring wall/API `22.673/18.464s`, cache load `1.532s`
- predicted official time: unavailable (requires fresh default panel)

## Next actions

- move work from online Activation quantization into calibration state
- inspect linear length=1024 first (mean delta=+0.000778, min delta=-0.000572)

> This tool never predicts an official score.
