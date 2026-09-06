# proxy-v3 diagnosis

**Decision: `continue_next_shard`**

## Warnings

- no official-time prediction: run a fresh default panel without calibration cache

## Accuracy localization

- linear: delta_mean `+0.000557`, L1 `0.002018`, delta_tail `+0.001075`, +/-/0 `33/23/0`
  - role_family `proj`: mean `-0.002340`, min `-0.010875`
  - role `proj`: mean `-0.002340`, min `-0.010875`
  - shape_bucket `wide_to_hidden`: mean `-0.002340`, min `-0.010875`
  - role_family `o`: mean `-0.001464`, min `-0.006933`

## Runtime localization

- `hif4_calibration_and_quantize_weight`: 44.850s (73.0%), 28 calls
- `hif4_dynamic_quantize_activation`: 16.587s (27.0%), 56 calls
- `hif4_calibration_attention`: 0.000s (0.0%), 0 calls
- `hif4_dynamic_quantize_q`: 0.000s (0.0%), 0 calls
- `hif4_dynamic_quantize_k`: 0.000s (0.0%), 0 calls
- `hif4_dynamic_quantize_v`: 0.000s (0.0%), 0 calls
- stages: calibration wall/API `46.072/44.850s`, scoring wall/API `20.858/16.587s`, cache load `0.000s`
- predicted official time: unavailable (requires fresh default panel)

## Next actions

- profile Weight calibration candidate loops, matrix factorizations, and repeated quantization
- inspect linear role_family=proj first (mean delta=-0.002340, min delta=-0.010875)

> This tool never predicts an official score.
