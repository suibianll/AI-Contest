# proxy-v3 diagnosis

**Decision: `continue_next_shard`**

## Warnings

- no official-time prediction: run a fresh default panel without calibration cache

## Accuracy localization

- linear: delta_mean `+0.000571`, L1 `0.002721`, delta_tail `+0.001123`, +/-/0 `30/26/0`
  - role `k`: mean `-0.001798`, min `-0.014397`
  - role_family `proj`: mean `-0.000903`, min `-0.007138`
  - role `proj`: mean `-0.000903`, min `-0.007138`
  - shape_bucket `wide_to_hidden`: mean `-0.000903`, min `-0.007138`

## Runtime localization

- `hif4_dynamic_quantize_activation`: 18.494s (100.0%), 56 calls
- `hif4_calibration_and_quantize_weight`: 0.000s (0.0%), 0 calls
- `hif4_calibration_attention`: 0.000s (0.0%), 0 calls
- `hif4_dynamic_quantize_q`: 0.000s (0.0%), 0 calls
- `hif4_dynamic_quantize_k`: 0.000s (0.0%), 0 calls
- `hif4_dynamic_quantize_v`: 0.000s (0.0%), 0 calls
- stages: calibration wall/API `0.000/0.000s`, scoring wall/API `23.115/18.494s`, cache load `1.433s`
- predicted official time: unavailable (requires fresh default panel)

## Next actions

- move work from online Activation quantization into calibration state
- inspect linear role=k first (mean delta=-0.001798, min delta=-0.014397)

> This tool never predicts an official score.
