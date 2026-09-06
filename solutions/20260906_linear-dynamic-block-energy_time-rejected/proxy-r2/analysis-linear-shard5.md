# proxy-v3 diagnosis

**Decision: `continue_next_shard`**

## Warnings

- no official-time prediction: run a fresh default panel without calibration cache

## Accuracy localization

- linear: delta_mean `+0.002926`, L1 `0.003030`, delta_tail `+0.002547`, +/-/0 `51/5/0`
  - role `k`: mean `+0.000687`, min `-0.000920`
  - role `fc_gate`: mean `+0.001302`, min `-0.000192`
  - role `q`: mean `+0.001317`, min `-0.001093`
  - role_family `fc`: mean `+0.001372`, min `-0.000192`

## Runtime localization

- `hif4_calibration_and_quantize_weight`: 46.177s (73.2%), 28 calls
- `hif4_dynamic_quantize_activation`: 16.919s (26.8%), 56 calls
- `hif4_calibration_attention`: 0.000s (0.0%), 0 calls
- `hif4_dynamic_quantize_q`: 0.000s (0.0%), 0 calls
- `hif4_dynamic_quantize_k`: 0.000s (0.0%), 0 calls
- `hif4_dynamic_quantize_v`: 0.000s (0.0%), 0 calls
- stages: calibration wall/API `47.414/46.177s`, scoring wall/API `21.396/16.919s`, cache load `0.000s`
- predicted official time: unavailable (requires fresh default panel)

## Next actions

- profile Weight calibration candidate loops, matrix factorizations, and repeated quantization
- inspect linear role=k first (mean delta=+0.000687, min delta=-0.000920)

> This tool never predicts an official score.
