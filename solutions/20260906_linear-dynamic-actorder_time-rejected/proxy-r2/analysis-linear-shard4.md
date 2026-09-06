# proxy-v3 diagnosis

**Decision: `continue_next_shard`**

## Warnings

- no official-time prediction: run a fresh default panel without calibration cache

## Accuracy localization

- linear: delta_mean `+0.002866`, L1 `0.003119`, delta_tail `+0.005181`, +/-/0 `49/7/0`
  - role `v`: mean `-0.000358`, min `-0.002610`
  - role_family `qkv`: mean `+0.000539`, min `-0.002610`
  - shape_bucket `hidden_to_wide`: mean `+0.000638`, min `-0.002610`
  - role `fc_up`: mean `+0.000680`, min `-0.000092`

## Runtime localization

- `hif4_calibration_and_quantize_weight`: 46.246s (73.1%), 28 calls
- `hif4_dynamic_quantize_activation`: 17.048s (26.9%), 56 calls
- `hif4_calibration_attention`: 0.000s (0.0%), 0 calls
- `hif4_dynamic_quantize_q`: 0.000s (0.0%), 0 calls
- `hif4_dynamic_quantize_k`: 0.000s (0.0%), 0 calls
- `hif4_dynamic_quantize_v`: 0.000s (0.0%), 0 calls
- stages: calibration wall/API `47.517/46.246s`, scoring wall/API `21.496/17.048s`, cache load `0.000s`
- predicted official time: unavailable (requires fresh default panel)

## Next actions

- profile Weight calibration candidate loops, matrix factorizations, and repeated quantization
- inspect linear role=v first (mean delta=-0.000358, min delta=-0.002610)

> This tool never predicts an official score.
