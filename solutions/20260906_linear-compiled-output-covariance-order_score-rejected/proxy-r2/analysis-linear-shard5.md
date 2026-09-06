# proxy-v3 diagnosis

**Decision: `continue_next_shard`**

## Warnings

- no official-time prediction: run a fresh default panel without calibration cache

## Accuracy localization

- linear: delta_mean `+0.000527`, L1 `0.000943`, delta_tail `+0.000517`, +/-/0 `33/13/10`
  - role_family `proj`: mean `+0.000000`, min `+0.000000`
  - role `proj`: mean `+0.000000`, min `+0.000000`
  - shape_bucket `wide_to_hidden`: mean `+0.000000`, min `+0.000000`
  - role_family `o`: mean `+0.000119`, min `-0.001881`

## Runtime localization

- `hif4_calibration_and_quantize_weight`: 44.713s (72.7%), 28 calls
- `hif4_dynamic_quantize_activation`: 16.768s (27.3%), 56 calls
- `hif4_calibration_attention`: 0.000s (0.0%), 0 calls
- `hif4_dynamic_quantize_q`: 0.000s (0.0%), 0 calls
- `hif4_dynamic_quantize_k`: 0.000s (0.0%), 0 calls
- `hif4_dynamic_quantize_v`: 0.000s (0.0%), 0 calls
- stages: calibration wall/API `45.909/44.713s`, scoring wall/API `20.675/16.768s`, cache load `0.000s`
- predicted official time: unavailable (requires fresh default panel)

## Next actions

- profile Weight calibration candidate loops, matrix factorizations, and repeated quantization
- inspect linear role_family=proj first (mean delta=+0.000000, min delta=+0.000000)

> This tool never predicts an official score.
