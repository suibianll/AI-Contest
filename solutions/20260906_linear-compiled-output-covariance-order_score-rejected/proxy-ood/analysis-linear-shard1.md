# proxy-v3 diagnosis

**Decision: `continue_next_shard`**

## Warnings

- linear: worst-20% tail regressed -0.000188
- no official-time prediction: run a fresh default panel without calibration cache

## Accuracy localization

- linear: delta_mean `+0.000465`, L1 `0.001501`, delta_tail `-0.000188`, +/-/0 `31/17/8`
  - layer `13`: mean `-0.000511`, min `-0.007557`
  - role `fc_up`: mean `-0.000328`, min `-0.004176`
  - role_family `proj`: mean `+0.000000`, min `+0.000000`
  - role `proj`: mean `+0.000000`, min `+0.000000`

## Runtime localization

- `hif4_dynamic_quantize_activation`: 18.260s (100.0%), 56 calls
- `hif4_calibration_and_quantize_weight`: 0.000s (0.0%), 0 calls
- `hif4_calibration_attention`: 0.000s (0.0%), 0 calls
- `hif4_dynamic_quantize_q`: 0.000s (0.0%), 0 calls
- `hif4_dynamic_quantize_k`: 0.000s (0.0%), 0 calls
- `hif4_dynamic_quantize_v`: 0.000s (0.0%), 0 calls
- stages: calibration wall/API `0.000/0.000s`, scoring wall/API `22.454/18.260s`, cache load `1.478s`
- predicted official time: unavailable (requires fresh default panel)

## Next actions

- move work from online Activation quantization into calibration state
- inspect linear layer=13 first (mean delta=-0.000511, min delta=-0.007557)

> This tool never predicts an official score.
