# proxy-v3 diagnosis

**Decision: `continue_next_shard`**

## Warnings

- no official-time prediction: run a fresh default panel without calibration cache

## Accuracy localization

- linear: delta_mean `+0.004704`, L1 `0.005402`, delta_tail `+0.013098`, +/-/0 `40/16/0`
  - role `v`: mean `-0.000927`, min `-0.002643`
  - role `fc_up`: mean `+0.000346`, min `-0.000853`
  - role_family `o`: mean `+0.001077`, min `-0.004688`
  - role `o`: mean `+0.001077`, min `-0.004688`

## Runtime localization

- `hif4_calibration_and_quantize_weight`: 45.842s (72.9%), 28 calls
- `hif4_dynamic_quantize_activation`: 17.084s (27.1%), 56 calls
- `hif4_calibration_attention`: 0.000s (0.0%), 0 calls
- `hif4_dynamic_quantize_q`: 0.000s (0.0%), 0 calls
- `hif4_dynamic_quantize_k`: 0.000s (0.0%), 0 calls
- `hif4_dynamic_quantize_v`: 0.000s (0.0%), 0 calls
- stages: calibration wall/API `47.061/45.842s`, scoring wall/API `21.439/17.084s`, cache load `0.000s`
- predicted official time: unavailable (requires fresh default panel)

## Next actions

- profile Weight calibration candidate loops, matrix factorizations, and repeated quantization
- inspect linear role=v first (mean delta=-0.000927, min delta=-0.002643)

> This tool never predicts an official score.
