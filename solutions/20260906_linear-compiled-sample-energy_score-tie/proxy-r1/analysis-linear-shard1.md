# proxy-v3 diagnosis

**Decision: `continue_next_shard`**

## Warnings

- no official-time prediction: run a fresh default panel without calibration cache

## Accuracy localization

- linear: delta_mean `+0.001949`, L1 `0.002170`, delta_tail `+0.000759`, +/-/0 `44/12/0`
  - role `fc_gate`: mean `+0.000539`, min `-0.000683`
  - role_family `fc`: mean `+0.000745`, min `-0.000683`
  - shape_bucket `hidden_to_wide`: mean `+0.000912`, min `-0.001493`
  - role `fc_up`: mean `+0.000951`, min `-0.000166`

## Runtime localization

- `hif4_calibration_and_quantize_weight`: 44.368s (72.2%), 28 calls
- `hif4_dynamic_quantize_activation`: 17.052s (27.8%), 56 calls
- `hif4_calibration_attention`: 0.000s (0.0%), 0 calls
- `hif4_dynamic_quantize_q`: 0.000s (0.0%), 0 calls
- `hif4_dynamic_quantize_k`: 0.000s (0.0%), 0 calls
- `hif4_dynamic_quantize_v`: 0.000s (0.0%), 0 calls
- stages: calibration wall/API `45.549/44.368s`, scoring wall/API `21.164/17.052s`, cache load `0.000s`
- predicted official time: unavailable (requires fresh default panel)

## Next actions

- profile Weight calibration candidate loops, matrix factorizations, and repeated quantization
- inspect linear role=fc_gate first (mean delta=+0.000539, min delta=-0.000683)

> This tool never predicts an official score.
