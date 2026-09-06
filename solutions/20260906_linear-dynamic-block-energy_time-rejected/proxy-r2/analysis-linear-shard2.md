# proxy-v3 diagnosis

**Decision: `continue_next_shard`**

## Warnings

- no official-time prediction: run a fresh default panel without calibration cache

## Accuracy localization

- linear: delta_mean `+0.006137`, L1 `0.006171`, delta_tail `+0.013868`, +/-/0 `53/3/0`
  - role `fc_up`: mean `+0.001112`, min `+0.000019`
  - role `v`: mean `+0.001242`, min `-0.000517`
  - role_family `fc`: mean `+0.002138`, min `-0.000407`
  - shape_bucket `hidden_to_wide`: mean `+0.002867`, min `-0.000517`

## Runtime localization

- `hif4_calibration_and_quantize_weight`: 45.771s (73.0%), 28 calls
- `hif4_dynamic_quantize_activation`: 16.922s (27.0%), 56 calls
- `hif4_calibration_attention`: 0.000s (0.0%), 0 calls
- `hif4_dynamic_quantize_q`: 0.000s (0.0%), 0 calls
- `hif4_dynamic_quantize_k`: 0.000s (0.0%), 0 calls
- `hif4_dynamic_quantize_v`: 0.000s (0.0%), 0 calls
- stages: calibration wall/API `46.956/45.771s`, scoring wall/API `21.343/16.922s`, cache load `0.000s`
- predicted official time: unavailable (requires fresh default panel)

## Next actions

- profile Weight calibration candidate loops, matrix factorizations, and repeated quantization
- inspect linear role=fc_up first (mean delta=+0.001112, min delta=+0.000019)

> This tool never predicts an official score.
