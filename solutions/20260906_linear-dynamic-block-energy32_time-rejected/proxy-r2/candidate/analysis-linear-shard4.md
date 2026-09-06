# proxy-v3 diagnosis

**Decision: `continue_next_shard`**

## Warnings

- no official-time prediction: run a fresh default panel without calibration cache

## Accuracy localization

- linear: delta_mean `+0.002597`, L1 `0.002735`, delta_tail `+0.004983`, +/-/0 `50/6/0`
  - role `v`: mean `+0.000507`, min `-0.001574`
  - role_family `qkv`: mean `+0.000829`, min `-0.001574`
  - role `q`: mean `+0.000858`, min `+0.000671`
  - shape_bucket `hidden_to_wide`: mean `+0.000953`, min `-0.001574`

## Runtime localization

- `hif4_calibration_and_quantize_weight`: 44.843s (72.2%), 28 calls
- `hif4_dynamic_quantize_activation`: 17.303s (27.8%), 56 calls
- `hif4_calibration_attention`: 0.000s (0.0%), 0 calls
- `hif4_dynamic_quantize_q`: 0.000s (0.0%), 0 calls
- `hif4_dynamic_quantize_k`: 0.000s (0.0%), 0 calls
- `hif4_dynamic_quantize_v`: 0.000s (0.0%), 0 calls
- stages: calibration wall/API `46.141/44.843s`, scoring wall/API `21.513/17.303s`, cache load `0.000s`
- predicted official time: unavailable (requires fresh default panel)

## Next actions

- profile Weight calibration candidate loops, matrix factorizations, and repeated quantization
- inspect linear role=v first (mean delta=+0.000507, min delta=-0.001574)

> This tool never predicts an official score.
