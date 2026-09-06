# proxy-v3 diagnosis

**Decision: `continue_next_shard`**

## Warnings

- no official-time prediction: run a fresh default panel without calibration cache

## Accuracy localization

- linear: delta_mean `+0.002868`, L1 `0.003047`, delta_tail `+0.002578`, +/-/0 `52/4/0`
  - role `k`: mean `+0.000571`, min `-0.001959`
  - role `q`: mean `+0.001091`, min `-0.001270`
  - role `fc_gate`: mean `+0.001278`, min `-0.001078`
  - role_family `fc`: mean `+0.001292`, min `-0.001078`

## Runtime localization

- `hif4_calibration_and_quantize_weight`: 44.995s (72.9%), 28 calls
- `hif4_dynamic_quantize_activation`: 16.767s (27.1%), 56 calls
- `hif4_calibration_attention`: 0.000s (0.0%), 0 calls
- `hif4_dynamic_quantize_q`: 0.000s (0.0%), 0 calls
- `hif4_dynamic_quantize_k`: 0.000s (0.0%), 0 calls
- `hif4_dynamic_quantize_v`: 0.000s (0.0%), 0 calls
- stages: calibration wall/API `46.185/44.995s`, scoring wall/API `20.705/16.767s`, cache load `0.000s`
- predicted official time: unavailable (requires fresh default panel)

## Next actions

- profile Weight calibration candidate loops, matrix factorizations, and repeated quantization
- inspect linear role=k first (mean delta=+0.000571, min delta=-0.001959)

> This tool never predicts an official score.
