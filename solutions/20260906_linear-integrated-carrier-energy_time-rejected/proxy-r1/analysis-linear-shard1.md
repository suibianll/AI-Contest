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

- `hif4_calibration_and_quantize_weight`: 45.920s (73.1%), 28 calls
- `hif4_dynamic_quantize_activation`: 16.857s (26.9%), 56 calls
- `hif4_calibration_attention`: 0.000s (0.0%), 0 calls
- `hif4_dynamic_quantize_q`: 0.000s (0.0%), 0 calls
- `hif4_dynamic_quantize_k`: 0.000s (0.0%), 0 calls
- `hif4_dynamic_quantize_v`: 0.000s (0.0%), 0 calls
- stages: calibration wall/API `47.102/45.920s`, scoring wall/API `20.818/16.857s`, cache load `0.000s`
- predicted official time: unavailable (requires fresh default panel)

## Next actions

- profile Weight calibration candidate loops, matrix factorizations, and repeated quantization
- inspect linear layer=1 first (mean delta=-0.000616, min delta=-0.004316)

> This tool never predicts an official score.
