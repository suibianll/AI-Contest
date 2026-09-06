# proxy-v3 diagnosis

**Decision: `continue_next_shard`**

## Warnings

- no official-time prediction: run a fresh default panel without calibration cache

## Accuracy localization

- linear: delta_mean `+0.003246`, L1 `0.003694`, delta_tail `+0.002330`, +/-/0 `47/9/0`
  - role `k`: mean `+0.000570`, min `-0.005933`
  - role `fc_gate`: mean `+0.000608`, min `-0.003369`
  - role_family `fc`: mean `+0.000789`, min `-0.003369`
  - role `fc_up`: mean `+0.000971`, min `-0.000138`

## Runtime localization

- `hif4_dynamic_quantize_activation`: 18.324s (100.0%), 56 calls
- `hif4_calibration_and_quantize_weight`: 0.000s (0.0%), 0 calls
- `hif4_calibration_attention`: 0.000s (0.0%), 0 calls
- `hif4_dynamic_quantize_q`: 0.000s (0.0%), 0 calls
- `hif4_dynamic_quantize_k`: 0.000s (0.0%), 0 calls
- `hif4_dynamic_quantize_v`: 0.000s (0.0%), 0 calls
- stages: calibration wall/API `0.000/0.000s`, scoring wall/API `22.951/18.324s`, cache load `1.479s`
- predicted official time: unavailable (requires fresh default panel)

## Next actions

- move work from online Activation quantization into calibration state
- inspect linear role=k first (mean delta=+0.000570, min delta=-0.005933)

> This tool never predicts an official score.
