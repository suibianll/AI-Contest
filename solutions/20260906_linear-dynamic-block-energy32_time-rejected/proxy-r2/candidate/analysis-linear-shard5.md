# proxy-v3 diagnosis

**Decision: `continue_next_shard`**

## Warnings

- no official-time prediction: run a fresh default panel without calibration cache

## Accuracy localization

- linear: delta_mean `+0.002358`, L1 `0.002550`, delta_tail `+0.001877`, +/-/0 `50/6/0`
  - role `q`: mean `+0.000826`, min `-0.001189`
  - layer `11`: mean `+0.001193`, min `-0.001303`
  - role `k`: mean `+0.001228`, min `-0.001214`
  - shape_bucket `hidden_to_hidden`: mean `+0.001303`, min `-0.001189`

## Runtime localization

- `hif4_calibration_and_quantize_weight`: 45.254s (72.4%), 28 calls
- `hif4_dynamic_quantize_activation`: 17.271s (27.6%), 56 calls
- `hif4_calibration_attention`: 0.000s (0.0%), 0 calls
- `hif4_dynamic_quantize_q`: 0.000s (0.0%), 0 calls
- `hif4_dynamic_quantize_k`: 0.000s (0.0%), 0 calls
- `hif4_dynamic_quantize_v`: 0.000s (0.0%), 0 calls
- stages: calibration wall/API `46.453/45.254s`, scoring wall/API `21.505/17.271s`, cache load `0.000s`
- predicted official time: unavailable (requires fresh default panel)

## Next actions

- profile Weight calibration candidate loops, matrix factorizations, and repeated quantization
- inspect linear role=q first (mean delta=+0.000826, min delta=-0.001189)

> This tool never predicts an official score.
