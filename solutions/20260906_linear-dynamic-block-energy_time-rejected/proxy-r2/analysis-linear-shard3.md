# proxy-v3 diagnosis

**Decision: `continue_next_shard`**

## Warnings

- no official-time prediction: run a fresh default panel without calibration cache

## Accuracy localization

- linear: delta_mean `+0.002561`, L1 `0.003072`, delta_tail `+0.002297`, +/-/0 `50/6/0`
  - role `q`: mean `+0.000353`, min `-0.000231`
  - role `k`: mean `+0.000579`, min `-0.002290`
  - layer `15`: mean `+0.001075`, min `-0.008782`
  - role `fc_gate`: mean `+0.001202`, min `-0.000049`

## Runtime localization

- `hif4_calibration_and_quantize_weight`: 45.681s (72.9%), 28 calls
- `hif4_dynamic_quantize_activation`: 16.940s (27.1%), 56 calls
- `hif4_calibration_attention`: 0.000s (0.0%), 0 calls
- `hif4_dynamic_quantize_q`: 0.000s (0.0%), 0 calls
- `hif4_dynamic_quantize_k`: 0.000s (0.0%), 0 calls
- `hif4_dynamic_quantize_v`: 0.000s (0.0%), 0 calls
- stages: calibration wall/API `46.865/45.681s`, scoring wall/API `21.472/16.940s`, cache load `0.000s`
- predicted official time: unavailable (requires fresh default panel)

## Next actions

- profile Weight calibration candidate loops, matrix factorizations, and repeated quantization
- inspect linear role=q first (mean delta=+0.000353, min delta=-0.000231)

> This tool never predicts an official score.
