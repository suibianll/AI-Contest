# proxy-v3 diagnosis

**Decision: `continue_next_shard`**

## Warnings

- no official-time prediction: run a fresh default panel without calibration cache

## Accuracy localization

- linear: delta_mean `+0.002860`, L1 `0.004173`, delta_tail `+0.006126`, +/-/0 `40/14/2`
  - role_family `o`: mean `-0.002036`, min `-0.008199`
  - role `o`: mean `-0.002036`, min `-0.008199`
  - shape_bucket `hidden_to_hidden`: mean `-0.000106`, min `-0.008199`
  - length `128`: mean `-0.000098`, min `-0.008199`

## Runtime localization

- `hif4_dynamic_quantize_activation`: 18.568s (100.0%), 56 calls
- `hif4_calibration_and_quantize_weight`: 0.000s (0.0%), 0 calls
- `hif4_calibration_attention`: 0.000s (0.0%), 0 calls
- `hif4_dynamic_quantize_q`: 0.000s (0.0%), 0 calls
- `hif4_dynamic_quantize_k`: 0.000s (0.0%), 0 calls
- `hif4_dynamic_quantize_v`: 0.000s (0.0%), 0 calls
- stages: calibration wall/API `0.000/0.000s`, scoring wall/API `23.311/18.568s`, cache load `1.481s`
- predicted official time: unavailable (requires fresh default panel)

## Next actions

- move work from online Activation quantization into calibration state
- inspect linear role_family=o first (mean delta=-0.002036, min delta=-0.008199)

> This tool never predicts an official score.
