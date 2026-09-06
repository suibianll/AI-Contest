# proxy-v3 diagnosis

**Decision: `continue_next_shard`**

## Warnings

- no official-time prediction: run a fresh default panel without calibration cache

## Accuracy localization

- linear: delta_mean `+0.005391`, L1 `0.005903`, delta_tail `+0.008120`, +/-/0 `48/8/0`
  - role `fc_up`: mean `+0.000649`, min `-0.003951`
  - layer `8`: mean `+0.001509`, min `-0.003951`
  - role `q`: mean `+0.001726`, min `-0.004986`
  - role_family `fc`: mean `+0.001734`, min `-0.003951`

## Runtime localization

- `hif4_dynamic_quantize_activation`: 19.648s (100.0%), 56 calls
- `hif4_calibration_and_quantize_weight`: 0.000s (0.0%), 0 calls
- `hif4_calibration_attention`: 0.000s (0.0%), 0 calls
- `hif4_dynamic_quantize_q`: 0.000s (0.0%), 0 calls
- `hif4_dynamic_quantize_k`: 0.000s (0.0%), 0 calls
- `hif4_dynamic_quantize_v`: 0.000s (0.0%), 0 calls
- stages: calibration wall/API `0.000/0.000s`, scoring wall/API `24.510/19.648s`, cache load `1.362s`
- predicted official time: unavailable (requires fresh default panel)

## Next actions

- move work from online Activation quantization into calibration state
- inspect linear role=fc_up first (mean delta=+0.000649, min delta=-0.003951)

> This tool never predicts an official score.
