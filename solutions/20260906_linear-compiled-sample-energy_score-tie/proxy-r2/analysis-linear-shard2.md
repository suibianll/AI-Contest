# proxy-v3 diagnosis

**Decision: `continue_next_shard`**

## Warnings

- no official-time prediction: run a fresh default panel without calibration cache

## Accuracy localization

- linear: delta_mean `+0.005743`, L1 `0.005791`, delta_tail `+0.013336`, +/-/0 `53/3/0`
  - role `fc_up`: mean `+0.000821`, min `-0.000463`
  - role `v`: mean `+0.001488`, min `-0.000584`
  - role_family `fc`: mean `+0.001936`, min `-0.000463`
  - role `q`: mean `+0.002322`, min `+0.000081`

## Runtime localization

- `hif4_calibration_and_quantize_weight`: 44.700s (72.6%), 28 calls
- `hif4_dynamic_quantize_activation`: 16.896s (27.4%), 56 calls
- `hif4_calibration_attention`: 0.000s (0.0%), 0 calls
- `hif4_dynamic_quantize_q`: 0.000s (0.0%), 0 calls
- `hif4_dynamic_quantize_k`: 0.000s (0.0%), 0 calls
- `hif4_dynamic_quantize_v`: 0.000s (0.0%), 0 calls
- stages: calibration wall/API `45.817/44.700s`, scoring wall/API `20.834/16.896s`, cache load `0.000s`
- predicted official time: unavailable (requires fresh default panel)

## Next actions

- profile Weight calibration candidate loops, matrix factorizations, and repeated quantization
- inspect linear role=fc_up first (mean delta=+0.000821, min delta=-0.000463)

> This tool never predicts an official score.
