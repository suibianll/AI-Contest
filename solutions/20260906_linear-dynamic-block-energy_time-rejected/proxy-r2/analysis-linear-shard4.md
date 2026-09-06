# proxy-v3 diagnosis

**Decision: `continue_next_shard`**

## Warnings

- no official-time prediction: run a fresh default panel without calibration cache

## Accuracy localization

- linear: delta_mean `+0.002865`, L1 `0.003276`, delta_tail `+0.005739`, +/-/0 `49/7/0`
  - role `v`: mean `-0.000776`, min `-0.004766`
  - shape_bucket `hidden_to_wide`: mean `+0.000522`, min `-0.004766`
  - role_family `qkv`: mean `+0.000637`, min `-0.004766`
  - role `fc_up`: mean `+0.000678`, min `-0.001022`

## Runtime localization

- `hif4_calibration_and_quantize_weight`: 46.010s (73.0%), 28 calls
- `hif4_dynamic_quantize_activation`: 16.975s (27.0%), 56 calls
- `hif4_calibration_attention`: 0.000s (0.0%), 0 calls
- `hif4_dynamic_quantize_q`: 0.000s (0.0%), 0 calls
- `hif4_dynamic_quantize_k`: 0.000s (0.0%), 0 calls
- `hif4_dynamic_quantize_v`: 0.000s (0.0%), 0 calls
- stages: calibration wall/API `47.223/46.010s`, scoring wall/API `21.448/16.975s`, cache load `0.000s`
- predicted official time: unavailable (requires fresh default panel)

## Next actions

- profile Weight calibration candidate loops, matrix factorizations, and repeated quantization
- inspect linear role=v first (mean delta=-0.000776, min delta=-0.004766)

> This tool never predicts an official score.
