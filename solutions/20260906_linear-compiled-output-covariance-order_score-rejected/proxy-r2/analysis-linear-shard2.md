# proxy-v3 diagnosis

**Decision: `continue_next_shard`**

## Warnings

- linear: worst-20% tail regressed -0.000016
- no official-time prediction: run a fresh default panel without calibration cache

## Accuracy localization

- linear: delta_mean `+0.001265`, L1 `0.001741`, delta_tail `-0.000016`, +/-/0 `35/13/8`
  - role_family `o`: mean `-0.000510`, min `-0.004309`
  - role `o`: mean `-0.000510`, min `-0.004309`
  - role_family `proj`: mean `+0.000000`, min `+0.000000`
  - role `proj`: mean `+0.000000`, min `+0.000000`

## Runtime localization

- `hif4_calibration_and_quantize_weight`: 44.418s (72.3%), 28 calls
- `hif4_dynamic_quantize_activation`: 16.994s (27.7%), 56 calls
- `hif4_calibration_attention`: 0.000s (0.0%), 0 calls
- `hif4_dynamic_quantize_q`: 0.000s (0.0%), 0 calls
- `hif4_dynamic_quantize_k`: 0.000s (0.0%), 0 calls
- `hif4_dynamic_quantize_v`: 0.000s (0.0%), 0 calls
- stages: calibration wall/API `45.589/44.418s`, scoring wall/API `20.926/16.994s`, cache load `0.000s`
- predicted official time: unavailable (requires fresh default panel)

## Next actions

- profile Weight calibration candidate loops, matrix factorizations, and repeated quantization
- inspect linear role_family=o first (mean delta=-0.000510, min delta=-0.004309)

> This tool never predicts an official score.
