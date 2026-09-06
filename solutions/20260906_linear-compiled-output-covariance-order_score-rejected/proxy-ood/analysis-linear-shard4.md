# proxy-v3 diagnosis

**Decision: `continue_next_shard`**

## Warnings

- no official-time prediction: run a fresh default panel without calibration cache

## Accuracy localization

- linear: delta_mean `+0.000592`, L1 `0.001653`, delta_tail `+0.000831`, +/-/0 `31/17/8`
  - layer `4`: mean `-0.000340`, min `-0.007177`
  - role `q`: mean `-0.000335`, min `-0.007177`
  - layer `16`: mean `-0.000295`, min `-0.006402`
  - length `128`: mean `-0.000020`, min `-0.006402`

## Runtime localization

- `hif4_dynamic_quantize_activation`: 18.241s (100.0%), 56 calls
- `hif4_calibration_and_quantize_weight`: 0.000s (0.0%), 0 calls
- `hif4_calibration_attention`: 0.000s (0.0%), 0 calls
- `hif4_dynamic_quantize_q`: 0.000s (0.0%), 0 calls
- `hif4_dynamic_quantize_k`: 0.000s (0.0%), 0 calls
- `hif4_dynamic_quantize_v`: 0.000s (0.0%), 0 calls
- stages: calibration wall/API `0.000/0.000s`, scoring wall/API `22.459/18.241s`, cache load `1.404s`
- predicted official time: unavailable (requires fresh default panel)

## Next actions

- move work from online Activation quantization into calibration state
- inspect linear layer=4 first (mean delta=-0.000340, min delta=-0.007177)

> This tool never predicts an official score.
