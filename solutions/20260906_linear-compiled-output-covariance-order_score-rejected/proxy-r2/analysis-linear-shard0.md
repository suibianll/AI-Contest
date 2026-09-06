# proxy-v3 diagnosis

**Decision: `continue_next_shard`**

## Warnings

- linear: worst-20% tail regressed -0.000749
- no official-time prediction: run a fresh default panel without calibration cache

## Accuracy localization

- linear: delta_mean `+0.000179`, L1 `0.001187`, delta_tail `-0.000749`, +/-/0 `27/21/8`
  - role_family `o`: mean `-0.000725`, min `-0.008691`
  - role `o`: mean `-0.000725`, min `-0.008691`
  - layer `0`: mean `-0.000452`, min `-0.002407`
  - shape_bucket `hidden_to_hidden`: mean `-0.000359`, min `-0.008691`

## Runtime localization

- `hif4_dynamic_quantize_activation`: 17.022s (100.0%), 56 calls
- `hif4_calibration_and_quantize_weight`: 0.000s (0.0%), 0 calls
- `hif4_calibration_attention`: 0.000s (0.0%), 0 calls
- `hif4_dynamic_quantize_q`: 0.000s (0.0%), 0 calls
- `hif4_dynamic_quantize_k`: 0.000s (0.0%), 0 calls
- `hif4_dynamic_quantize_v`: 0.000s (0.0%), 0 calls
- stages: calibration wall/API `0.000/0.000s`, scoring wall/API `20.967/17.022s`, cache load `1.401s`
- predicted official time: unavailable (requires fresh default panel)

## Next actions

- move work from online Activation quantization into calibration state
- inspect linear role_family=o first (mean delta=-0.000725, min delta=-0.008691)

> This tool never predicts an official score.
