# proxy-v3 diagnosis

**Decision: `continue_next_shard`**

## Warnings

- linear: worst-20% tail regressed -0.000168
- no official-time prediction: run a fresh default panel without calibration cache

## Accuracy localization

- linear: delta_mean `+0.000272`, L1 `0.001277`, delta_tail `-0.000168`, +/-/0 `38/18/0`
  - layer `4`: mean `-0.000752`, min `-0.004788`
  - role_family `o`: mean `-0.000350`, min `-0.004788`
  - role `o`: mean `-0.000350`, min `-0.004788`
  - length `128`: mean `-0.000117`, min `-0.004528`

## Runtime localization

- `hif4_calibration_and_quantize_weight`: 45.331s (72.9%), 28 calls
- `hif4_dynamic_quantize_activation`: 16.827s (27.1%), 56 calls
- `hif4_calibration_attention`: 0.000s (0.0%), 0 calls
- `hif4_dynamic_quantize_q`: 0.000s (0.0%), 0 calls
- `hif4_dynamic_quantize_k`: 0.000s (0.0%), 0 calls
- `hif4_dynamic_quantize_v`: 0.000s (0.0%), 0 calls
- stages: calibration wall/API `46.567/45.331s`, scoring wall/API `21.075/16.827s`, cache load `0.000s`
- predicted official time: unavailable (requires fresh default panel)

## Next actions

- profile Weight calibration candidate loops, matrix factorizations, and repeated quantization
- inspect linear layer=4 first (mean delta=-0.000752, min delta=-0.004788)

> This tool never predicts an official score.
