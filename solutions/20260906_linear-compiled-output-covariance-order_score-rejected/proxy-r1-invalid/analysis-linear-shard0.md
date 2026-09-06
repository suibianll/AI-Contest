# proxy-v3 diagnosis

**Decision: `reject`**

## Blockers

- linear: delta_mean=-0.000365 <= 0

## Warnings

- linear: worst-20% tail regressed -0.000802
- no official-time prediction: run a fresh default panel without calibration cache

## Accuracy localization

- linear: delta_mean `-0.000365`, L1 `0.001731`, delta_tail `-0.000802`, +/-/0 `27/29/0`
  - role_family `proj`: mean `-0.003804`, min `-0.007417`
  - role `proj`: mean `-0.003804`, min `-0.007417`
  - shape_bucket `wide_to_hidden`: mean `-0.003804`, min `-0.007417`
  - layer `0`: mean `-0.001023`, min `-0.007417`

## Runtime localization

- `hif4_calibration_and_quantize_weight`: 45.826s (73.1%), 28 calls
- `hif4_dynamic_quantize_activation`: 16.855s (26.9%), 56 calls
- `hif4_calibration_attention`: 0.000s (0.0%), 0 calls
- `hif4_dynamic_quantize_q`: 0.000s (0.0%), 0 calls
- `hif4_dynamic_quantize_k`: 0.000s (0.0%), 0 calls
- `hif4_dynamic_quantize_v`: 0.000s (0.0%), 0 calls
- stages: calibration wall/API `47.012/45.826s`, scoring wall/API `20.830/16.855s`, cache load `0.000s`
- predicted official time: unavailable (requires fresh default panel)

## Next actions

- profile Weight calibration candidate loops, matrix factorizations, and repeated quantization
- inspect linear role_family=proj first (mean delta=-0.003804, min delta=-0.007417)

> This tool never predicts an official score.
