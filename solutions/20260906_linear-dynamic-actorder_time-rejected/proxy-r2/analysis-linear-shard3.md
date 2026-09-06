# proxy-v3 diagnosis

**Decision: `continue_next_shard`**

## Warnings

- no official-time prediction: run a fresh default panel without calibration cache

## Accuracy localization

- linear: delta_mean `+0.002554`, L1 `0.003050`, delta_tail `+0.002458`, +/-/0 `50/6/0`
  - role `q`: mean `+0.000329`, min `-0.000535`
  - layer `15`: mean `+0.000889`, min `-0.008782`
  - role `fc_gate`: mean `+0.001211`, min `-0.000213`
  - role `k`: mean `+0.001326`, min `-0.001527`

## Runtime localization

- `hif4_calibration_and_quantize_weight`: 45.983s (73.3%), 28 calls
- `hif4_dynamic_quantize_activation`: 16.774s (26.7%), 56 calls
- `hif4_calibration_attention`: 0.000s (0.0%), 0 calls
- `hif4_dynamic_quantize_q`: 0.000s (0.0%), 0 calls
- `hif4_dynamic_quantize_k`: 0.000s (0.0%), 0 calls
- `hif4_dynamic_quantize_v`: 0.000s (0.0%), 0 calls
- stages: calibration wall/API `47.232/45.983s`, scoring wall/API `21.292/16.774s`, cache load `0.000s`
- predicted official time: unavailable (requires fresh default panel)

## Next actions

- profile Weight calibration candidate loops, matrix factorizations, and repeated quantization
- inspect linear role=q first (mean delta=+0.000329, min delta=-0.000535)

> This tool never predicts an official score.
