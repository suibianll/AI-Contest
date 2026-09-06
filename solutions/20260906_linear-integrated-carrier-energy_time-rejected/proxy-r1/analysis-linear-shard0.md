# proxy-v3 diagnosis

**Decision: `continue_next_shard`**

## Warnings

- linear: worst-20% tail regressed -0.000729
- no official-time prediction: run a fresh default panel without calibration cache

## Accuracy localization

- linear: delta_mean `+0.000207`, L1 `0.001452`, delta_tail `-0.000729`, +/-/0 `36/20/0`
  - role_family `o`: mean `-0.001429`, min `-0.009218`
  - role `o`: mean `-0.001429`, min `-0.009218`
  - shape_bucket `hidden_to_hidden`: mean `-0.000847`, min `-0.009218`
  - layer `0`: mean `-0.000656`, min `-0.002313`

## Runtime localization

- `hif4_calibration_and_quantize_weight`: 48.361s (73.2%), 28 calls
- `hif4_dynamic_quantize_activation`: 17.743s (26.8%), 56 calls
- `hif4_calibration_attention`: 0.000s (0.0%), 0 calls
- `hif4_dynamic_quantize_q`: 0.000s (0.0%), 0 calls
- `hif4_dynamic_quantize_k`: 0.000s (0.0%), 0 calls
- `hif4_dynamic_quantize_v`: 0.000s (0.0%), 0 calls
- stages: calibration wall/API `49.621/48.361s`, scoring wall/API `21.936/17.743s`, cache load `0.000s`
- predicted official time: unavailable (requires fresh default panel)

## Next actions

- profile Weight calibration candidate loops, matrix factorizations, and repeated quantization
- inspect linear role_family=o first (mean delta=-0.001429, min delta=-0.009218)

> This tool never predicts an official score.
