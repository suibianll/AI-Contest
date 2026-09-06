# proxy-v3 diagnosis

**Decision: `continue_next_shard`**

## Warnings

- linear: worst-20% tail regressed -0.000264
- no official-time prediction: run a fresh default panel without calibration cache

## Accuracy localization

- linear: delta_mean `+0.000191`, L1 `0.001269`, delta_tail `-0.000264`, +/-/0 `26/26/4`
  - role_family `proj`: mean `-0.000762`, min `-0.002180`
  - role `proj`: mean `-0.000762`, min `-0.002180`
  - shape_bucket `wide_to_hidden`: mean `-0.000762`, min `-0.002180`
  - layer `1`: mean `-0.000644`, min `-0.003457`

## Runtime localization

- `hif4_calibration_and_quantize_weight`: 46.214s (73.1%), 28 calls
- `hif4_dynamic_quantize_activation`: 16.983s (26.9%), 56 calls
- `hif4_calibration_attention`: 0.000s (0.0%), 0 calls
- `hif4_dynamic_quantize_q`: 0.000s (0.0%), 0 calls
- `hif4_dynamic_quantize_k`: 0.000s (0.0%), 0 calls
- `hif4_dynamic_quantize_v`: 0.000s (0.0%), 0 calls
- stages: calibration wall/API `47.414/46.214s`, scoring wall/API `21.539/16.983s`, cache load `0.000s`
- predicted official time: unavailable (requires fresh default panel)

## Next actions

- profile Weight calibration candidate loops, matrix factorizations, and repeated quantization
- inspect linear role_family=proj first (mean delta=-0.000762, min delta=-0.002180)

> This tool never predicts an official score.
