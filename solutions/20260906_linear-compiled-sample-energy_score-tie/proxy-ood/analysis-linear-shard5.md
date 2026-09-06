# proxy-v3 diagnosis

**Decision: `continue_next_shard`**

## Warnings

- no official-time prediction: run a fresh default panel without calibration cache

## Accuracy localization

- linear: delta_mean `+0.003046`, L1 `0.003375`, delta_tail `+0.001887`, +/-/0 `48/8/0`
  - role `k`: mean `+0.000685`, min `-0.000989`
  - role `fc_up`: mean `+0.000858`, min `+0.000004`
  - role_family `o`: mean `+0.001051`, min `-0.005757`
  - role `o`: mean `+0.001051`, min `-0.005757`

## Runtime localization

- `hif4_dynamic_quantize_activation`: 18.382s (100.0%), 56 calls
- `hif4_calibration_and_quantize_weight`: 0.000s (0.0%), 0 calls
- `hif4_calibration_attention`: 0.000s (0.0%), 0 calls
- `hif4_dynamic_quantize_q`: 0.000s (0.0%), 0 calls
- `hif4_dynamic_quantize_k`: 0.000s (0.0%), 0 calls
- `hif4_dynamic_quantize_v`: 0.000s (0.0%), 0 calls
- stages: calibration wall/API `0.000/0.000s`, scoring wall/API `22.502/18.382s`, cache load `1.523s`
- predicted official time: unavailable (requires fresh default panel)

## Next actions

- move work from online Activation quantization into calibration state
- inspect linear role=k first (mean delta=+0.000685, min delta=-0.000989)

> This tool never predicts an official score.
