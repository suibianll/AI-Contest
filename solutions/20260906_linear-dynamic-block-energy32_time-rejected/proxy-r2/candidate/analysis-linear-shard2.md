# proxy-v3 diagnosis

**Decision: `continue_next_shard`**

## Warnings

- no official-time prediction: run a fresh default panel without calibration cache

## Accuracy localization

- linear: delta_mean `+0.005471`, L1 `0.005642`, delta_tail `+0.013234`, +/-/0 `47/9/0`
  - role `v`: mean `+0.000205`, min `-0.000689`
  - role `fc_up`: mean `+0.000717`, min `-0.001035`
  - role_family `fc`: mean `+0.001672`, min `-0.001035`
  - shape_bucket `hidden_to_wide`: mean `+0.002230`, min `-0.001035`

## Runtime localization

- `hif4_calibration_and_quantize_weight`: 46.068s (73.0%), 28 calls
- `hif4_dynamic_quantize_activation`: 17.057s (27.0%), 56 calls
- `hif4_calibration_attention`: 0.000s (0.0%), 0 calls
- `hif4_dynamic_quantize_q`: 0.000s (0.0%), 0 calls
- `hif4_dynamic_quantize_k`: 0.000s (0.0%), 0 calls
- `hif4_dynamic_quantize_v`: 0.000s (0.0%), 0 calls
- stages: calibration wall/API `47.293/46.068s`, scoring wall/API `21.555/17.057s`, cache load `0.000s`
- predicted official time: unavailable (requires fresh default panel)

## Next actions

- profile Weight calibration candidate loops, matrix factorizations, and repeated quantization
- inspect linear role=v first (mean delta=+0.000205, min delta=-0.000689)

> This tool never predicts an official score.
