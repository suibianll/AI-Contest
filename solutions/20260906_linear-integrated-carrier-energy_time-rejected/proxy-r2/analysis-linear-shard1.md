# proxy-v3 diagnosis

**Decision: `continue_next_shard`**

## Warnings

- linear: worst-20% tail regressed -0.000384
- no official-time prediction: run a fresh default panel without calibration cache

## Accuracy localization

- linear: delta_mean `+0.000201`, L1 `0.001084`, delta_tail `-0.000384`, +/-/0 `31/23/2`
  - layer `1`: mean `-0.000616`, min `-0.004316`
  - role `k`: mean `-0.000288`, min `-0.004316`
  - role_family `proj`: mean `-0.000199`, min `-0.003458`
  - role `proj`: mean `-0.000199`, min `-0.003458`

## Runtime localization

- `hif4_dynamic_quantize_activation`: 17.547s (100.0%), 56 calls
- `hif4_calibration_and_quantize_weight`: 0.000s (0.0%), 0 calls
- `hif4_calibration_attention`: 0.000s (0.0%), 0 calls
- `hif4_dynamic_quantize_q`: 0.000s (0.0%), 0 calls
- `hif4_dynamic_quantize_k`: 0.000s (0.0%), 0 calls
- `hif4_dynamic_quantize_v`: 0.000s (0.0%), 0 calls
- stages: calibration wall/API `0.000/0.000s`, scoring wall/API `22.040/17.547s`, cache load `1.392s`
- predicted official time: unavailable (requires fresh default panel)

## Next actions

- move work from online Activation quantization into calibration state
- inspect linear layer=1 first (mean delta=-0.000616, min delta=-0.004316)

> This tool never predicts an official score.
