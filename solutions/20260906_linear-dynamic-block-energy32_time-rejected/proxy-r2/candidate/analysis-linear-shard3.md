# proxy-v3 diagnosis

**Decision: `continue_next_shard`**

## Warnings

- no official-time prediction: run a fresh default panel without calibration cache

## Accuracy localization

- linear: delta_mean `+0.002273`, L1 `0.002713`, delta_tail `+0.002297`, +/-/0 `46/10/0`
  - role `q`: mean `+0.000547`, min `-0.000638`
  - role `fc_gate`: mean `+0.000902`, min `-0.000060`
  - role `k`: mean `+0.001190`, min `-0.001836`
  - layer `15`: mean `+0.001310`, min `-0.006340`

## Runtime localization

- `hif4_calibration_and_quantize_weight`: 46.667s (73.0%), 28 calls
- `hif4_dynamic_quantize_activation`: 17.236s (27.0%), 56 calls
- `hif4_calibration_attention`: 0.000s (0.0%), 0 calls
- `hif4_dynamic_quantize_q`: 0.000s (0.0%), 0 calls
- `hif4_dynamic_quantize_k`: 0.000s (0.0%), 0 calls
- `hif4_dynamic_quantize_v`: 0.000s (0.0%), 0 calls
- stages: calibration wall/API `47.916/46.667s`, scoring wall/API `21.491/17.236s`, cache load `0.000s`
- predicted official time: unavailable (requires fresh default panel)

## Next actions

- profile Weight calibration candidate loops, matrix factorizations, and repeated quantization
- inspect linear role=q first (mean delta=+0.000547, min delta=-0.000638)

> This tool never predicts an official score.
