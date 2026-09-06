# proxy-v3 diagnosis

**Decision: `continue_next_shard`**

## Warnings

- no official-time prediction: run a fresh default panel without calibration cache

## Accuracy localization

- linear: delta_mean `+0.000016`, L1 `0.002557`, delta_tail `+0.001282`, +/-/0 `33/23/0`
  - role `k`: mean `-0.001803`, min `-0.010630`
  - layer `18`: mean `-0.001345`, min `-0.010630`
  - role `q`: mean `-0.001123`, min `-0.005017`
  - layer `0`: mean `-0.000767`, min `-0.005017`

## Runtime localization

- `hif4_dynamic_quantize_activation`: 18.204s (100.0%), 56 calls
- `hif4_calibration_and_quantize_weight`: 0.000s (0.0%), 0 calls
- `hif4_calibration_attention`: 0.000s (0.0%), 0 calls
- `hif4_dynamic_quantize_q`: 0.000s (0.0%), 0 calls
- `hif4_dynamic_quantize_k`: 0.000s (0.0%), 0 calls
- `hif4_dynamic_quantize_v`: 0.000s (0.0%), 0 calls
- stages: calibration wall/API `0.000/0.000s`, scoring wall/API `22.804/18.204s`, cache load `1.493s`
- predicted official time: unavailable (requires fresh default panel)

## Next actions

- move work from online Activation quantization into calibration state
- inspect linear role=k first (mean delta=-0.001803, min delta=-0.010630)

> This tool never predicts an official score.
