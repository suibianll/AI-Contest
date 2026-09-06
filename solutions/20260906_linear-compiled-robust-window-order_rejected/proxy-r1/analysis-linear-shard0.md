# proxy-v3 diagnosis

**Decision: `reject`**

## Blockers

- linear: delta_mean=-0.000043 <= 0

## Warnings

- linear: worst-20% tail regressed -0.001098
- no official-time prediction: run a fresh default panel without calibration cache

## Accuracy localization

- linear: delta_mean `-0.000043`, L1 `0.001434`, delta_tail `-0.001098`, +/-/0 `26/28/2`
  - role_family `o`: mean `-0.002016`, min `-0.012236`
  - role `o`: mean `-0.002016`, min `-0.012236`
  - shape_bucket `hidden_to_hidden`: mean `-0.001228`, min `-0.012236`
  - layer `0`: mean `-0.001022`, min `-0.002430`

## Runtime localization

- `hif4_calibration_and_quantize_weight`: 46.932s (73.2%), 28 calls
- `hif4_dynamic_quantize_activation`: 17.172s (26.8%), 56 calls
- `hif4_calibration_attention`: 0.000s (0.0%), 0 calls
- `hif4_dynamic_quantize_q`: 0.000s (0.0%), 0 calls
- `hif4_dynamic_quantize_k`: 0.000s (0.0%), 0 calls
- `hif4_dynamic_quantize_v`: 0.000s (0.0%), 0 calls
- stages: calibration wall/API `48.192/46.932s`, scoring wall/API `21.752/17.172s`, cache load `0.000s`
- predicted official time: unavailable (requires fresh default panel)

## Next actions

- profile Weight calibration candidate loops, matrix factorizations, and repeated quantization
- inspect linear role_family=o first (mean delta=-0.002016, min delta=-0.012236)

> This tool never predicts an official score.
