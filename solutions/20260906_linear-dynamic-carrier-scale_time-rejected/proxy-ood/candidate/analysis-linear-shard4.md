# proxy-v3 diagnosis

**Decision: `continue_next_shard`**

## Warnings

- no official-time prediction: run a fresh default panel without calibration cache

## Accuracy localization

- linear: delta_mean `+0.002052`, L1 `0.003764`, delta_tail `+0.001631`, +/-/0 `43/13/0`
  - layer `10`: mean `-0.001082`, min `-0.012710`
  - role `k`: mean `-0.000974`, min `-0.009437`
  - role `fc_gate`: mean `+0.000362`, min `-0.002971`
  - role_family `o`: mean `+0.000586`, min `-0.012710`

## Runtime localization

- `hif4_dynamic_quantize_activation`: 19.925s (100.0%), 56 calls
- `hif4_calibration_and_quantize_weight`: 0.000s (0.0%), 0 calls
- `hif4_calibration_attention`: 0.000s (0.0%), 0 calls
- `hif4_dynamic_quantize_q`: 0.000s (0.0%), 0 calls
- `hif4_dynamic_quantize_k`: 0.000s (0.0%), 0 calls
- `hif4_dynamic_quantize_v`: 0.000s (0.0%), 0 calls
- stages: calibration wall/API `0.000/0.000s`, scoring wall/API `24.864/19.925s`, cache load `1.611s`
- predicted official time: unavailable (requires fresh default panel)

## Next actions

- move work from online Activation quantization into calibration state
- inspect linear layer=10 first (mean delta=-0.001082, min delta=-0.012710)

> This tool never predicts an official score.
