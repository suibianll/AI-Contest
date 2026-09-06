# proxy-v3 diagnosis

**Decision: `continue_next_shard`**

## Warnings

- no official-time prediction: run a fresh default panel without calibration cache

## Accuracy localization

- linear: delta_mean `+0.001045`, L1 `0.001507`, delta_tail `+0.000867`, +/-/0 `41/15/0`
  - role `k`: mean `+0.000252`, min `-0.001978`
  - role_family `o`: mean `+0.000289`, min `-0.001421`
  - role `o`: mean `+0.000289`, min `-0.001421`
  - shape_bucket `hidden_to_hidden`: mean `+0.000437`, min `-0.001421`

## Runtime localization

- `hif4_calibration_and_quantize_weight`: 45.871s (73.3%), 28 calls
- `hif4_dynamic_quantize_activation`: 16.716s (26.7%), 56 calls
- `hif4_calibration_attention`: 0.000s (0.0%), 0 calls
- `hif4_dynamic_quantize_q`: 0.000s (0.0%), 0 calls
- `hif4_dynamic_quantize_k`: 0.000s (0.0%), 0 calls
- `hif4_dynamic_quantize_v`: 0.000s (0.0%), 0 calls
- stages: calibration wall/API `47.070/45.871s`, scoring wall/API `21.029/16.716s`, cache load `0.000s`
- predicted official time: unavailable (requires fresh default panel)

## Next actions

- profile Weight calibration candidate loops, matrix factorizations, and repeated quantization
- inspect linear role=k first (mean delta=+0.000252, min delta=-0.001978)

> This tool never predicts an official score.
