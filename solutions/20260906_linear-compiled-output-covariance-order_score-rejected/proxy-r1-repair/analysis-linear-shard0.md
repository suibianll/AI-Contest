# proxy-v3 diagnosis

**Decision: `continue_next_shard`**

## Warnings

- linear: worst-20% tail regressed -0.000749
- no official-time prediction: run a fresh default panel without calibration cache

## Accuracy localization

- linear: delta_mean `+0.000179`, L1 `0.001187`, delta_tail `-0.000749`, +/-/0 `27/21/8`
  - role_family `o`: mean `-0.000725`, min `-0.008691`
  - role `o`: mean `-0.000725`, min `-0.008691`
  - layer `0`: mean `-0.000452`, min `-0.002407`
  - shape_bucket `hidden_to_hidden`: mean `-0.000359`, min `-0.008691`

## Runtime localization

- `hif4_calibration_and_quantize_weight`: 45.339s (72.7%), 28 calls
- `hif4_dynamic_quantize_activation`: 16.984s (27.3%), 56 calls
- `hif4_calibration_attention`: 0.000s (0.0%), 0 calls
- `hif4_dynamic_quantize_q`: 0.000s (0.0%), 0 calls
- `hif4_dynamic_quantize_k`: 0.000s (0.0%), 0 calls
- `hif4_dynamic_quantize_v`: 0.000s (0.0%), 0 calls
- stages: calibration wall/API `46.560/45.339s`, scoring wall/API `21.018/16.984s`, cache load `0.000s`
- predicted official time: unavailable (requires fresh default panel)

## Next actions

- profile Weight calibration candidate loops, matrix factorizations, and repeated quantization
- inspect linear role_family=o first (mean delta=-0.000725, min delta=-0.008691)

> This tool never predicts an official score.
