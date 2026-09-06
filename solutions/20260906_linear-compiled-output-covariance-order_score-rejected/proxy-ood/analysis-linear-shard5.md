# proxy-v3 diagnosis

**Decision: `continue_next_shard`**

## Warnings

- no official-time prediction: run a fresh default panel without calibration cache

## Accuracy localization

- linear: delta_mean `+0.000927`, L1 `0.001620`, delta_tail `+0.001342`, +/-/0 `34/12/10`
  - role_family `o`: mean `-0.001123`, min `-0.007233`
  - role `o`: mean `-0.001123`, min `-0.007233`
  - shape_bucket `hidden_to_hidden`: mean `-0.000197`, min `-0.007233`
  - role_family `proj`: mean `+0.000000`, min `+0.000000`

## Runtime localization

- `hif4_dynamic_quantize_activation`: 18.269s (100.0%), 56 calls
- `hif4_calibration_and_quantize_weight`: 0.000s (0.0%), 0 calls
- `hif4_calibration_attention`: 0.000s (0.0%), 0 calls
- `hif4_dynamic_quantize_q`: 0.000s (0.0%), 0 calls
- `hif4_dynamic_quantize_k`: 0.000s (0.0%), 0 calls
- `hif4_dynamic_quantize_v`: 0.000s (0.0%), 0 calls
- stages: calibration wall/API `0.000/0.000s`, scoring wall/API `22.492/18.269s`, cache load `1.516s`
- predicted official time: unavailable (requires fresh default panel)

## Next actions

- move work from online Activation quantization into calibration state
- inspect linear role_family=o first (mean delta=-0.001123, min delta=-0.007233)

> This tool never predicts an official score.
