# proxy-v3 diagnosis

**Decision: `continue_next_shard`**

## Warnings

- no official-time prediction: run a fresh default panel without calibration cache

## Accuracy localization

- linear: delta_mean `+0.000512`, L1 `0.002246`, delta_tail `+0.000429`, +/-/0 `41/15/0`
  - role `k`: mean `-0.000583`, min `-0.013453`
  - length `10`: mean `-0.000332`, min `-0.013453`
  - layer `4`: mean `-0.000198`, min `-0.003889`
  - role `v`: mean `-0.000151`, min `-0.003124`

## Runtime localization

- `hif4_dynamic_quantize_activation`: 17.974s (100.0%), 56 calls
- `hif4_calibration_and_quantize_weight`: 0.000s (0.0%), 0 calls
- `hif4_calibration_attention`: 0.000s (0.0%), 0 calls
- `hif4_dynamic_quantize_q`: 0.000s (0.0%), 0 calls
- `hif4_dynamic_quantize_k`: 0.000s (0.0%), 0 calls
- `hif4_dynamic_quantize_v`: 0.000s (0.0%), 0 calls
- stages: calibration wall/API `0.000/0.000s`, scoring wall/API `22.569/17.974s`, cache load `1.427s`
- predicted official time: unavailable (requires fresh default panel)

## Next actions

- move work from online Activation quantization into calibration state
- inspect linear role=k first (mean delta=-0.000583, min delta=-0.013453)

> This tool never predicts an official score.
