# proxy-v3 diagnosis

**Decision: `continue_next_shard`**

## Warnings

- no official-time prediction: run a fresh default panel without calibration cache

## Accuracy localization

- linear: delta_mean `+0.001238`, L1 `0.001687`, delta_tail `+0.000123`, +/-/0 `41/15/0`
  - role `v`: mean `+0.000444`, min `-0.001441`
  - layer `1`: mean `+0.000462`, min `-0.001879`
  - role `fc_gate`: mean `+0.000465`, min `-0.000615`
  - role `k`: mean `+0.000471`, min `-0.001879`

## Runtime localization

- `hif4_dynamic_quantize_activation`: 17.076s (100.0%), 56 calls
- `hif4_calibration_and_quantize_weight`: 0.000s (0.0%), 0 calls
- `hif4_calibration_attention`: 0.000s (0.0%), 0 calls
- `hif4_dynamic_quantize_q`: 0.000s (0.0%), 0 calls
- `hif4_dynamic_quantize_k`: 0.000s (0.0%), 0 calls
- `hif4_dynamic_quantize_v`: 0.000s (0.0%), 0 calls
- stages: calibration wall/API `0.000/0.000s`, scoring wall/API `21.446/17.076s`, cache load `1.595s`
- predicted official time: unavailable (requires fresh default panel)

## Next actions

- move work from online Activation quantization into calibration state
- inspect linear role=v first (mean delta=+0.000444, min delta=-0.001441)

> This tool never predicts an official score.
