# proxy-v3 diagnosis

**Decision: `continue_next_shard`**

## Warnings

- no official-time prediction: run a fresh default panel without calibration cache

## Accuracy localization

- linear: delta_mean `+0.000354`, L1 `0.001042`, delta_tail `+0.000777`, +/-/0 `30/18/8`
  - role `v`: mean `-0.000362`, min `-0.004038`
  - layer `10`: mean `-0.000275`, min `-0.001961`
  - role_family `proj`: mean `+0.000000`, min `+0.000000`
  - role `proj`: mean `+0.000000`, min `+0.000000`

## Runtime localization

- `hif4_calibration_and_quantize_weight`: 44.286s (72.3%), 28 calls
- `hif4_dynamic_quantize_activation`: 16.945s (27.7%), 56 calls
- `hif4_calibration_attention`: 0.000s (0.0%), 0 calls
- `hif4_dynamic_quantize_q`: 0.000s (0.0%), 0 calls
- `hif4_dynamic_quantize_k`: 0.000s (0.0%), 0 calls
- `hif4_dynamic_quantize_v`: 0.000s (0.0%), 0 calls
- stages: calibration wall/API `45.562/44.286s`, scoring wall/API `20.909/16.945s`, cache load `0.000s`
- predicted official time: unavailable (requires fresh default panel)

## Next actions

- profile Weight calibration candidate loops, matrix factorizations, and repeated quantization
- inspect linear role=v first (mean delta=-0.000362, min delta=-0.004038)

> This tool never predicts an official score.
