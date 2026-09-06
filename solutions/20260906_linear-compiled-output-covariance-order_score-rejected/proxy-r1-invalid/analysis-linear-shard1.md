# proxy-v3 diagnosis

**Decision: `reject`**

## Blockers

- linear: delta_mean=-0.000123 <= 0

## Warnings

- linear: worst-20% tail regressed -0.000111
- no official-time prediction: run a fresh default panel without calibration cache

## Accuracy localization

- linear: delta_mean `-0.000123`, L1 `0.001508`, delta_tail `-0.000111`, +/-/0 `32/24/0`
  - role_family `proj`: mean `-0.002808`, min `-0.005866`
  - role `proj`: mean `-0.002808`, min `-0.005866`
  - shape_bucket `wide_to_hidden`: mean `-0.002808`, min `-0.005866`
  - layer `1`: mean `-0.000758`, min `-0.004702`

## Runtime localization

- `hif4_calibration_and_quantize_weight`: 44.904s (72.6%), 28 calls
- `hif4_dynamic_quantize_activation`: 16.987s (27.4%), 56 calls
- `hif4_calibration_attention`: 0.000s (0.0%), 0 calls
- `hif4_dynamic_quantize_q`: 0.000s (0.0%), 0 calls
- `hif4_dynamic_quantize_k`: 0.000s (0.0%), 0 calls
- `hif4_dynamic_quantize_v`: 0.000s (0.0%), 0 calls
- stages: calibration wall/API `46.117/44.904s`, scoring wall/API `21.026/16.987s`, cache load `0.000s`
- predicted official time: unavailable (requires fresh default panel)

## Next actions

- profile Weight calibration candidate loops, matrix factorizations, and repeated quantization
- inspect linear role_family=proj first (mean delta=-0.002808, min delta=-0.005866)

> This tool never predicts an official score.
