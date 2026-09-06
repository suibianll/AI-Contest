# proxy-v3 diagnosis

**Decision: `continue_next_shard`**

## Warnings

- no official-time prediction: run a fresh default panel without calibration cache

## Accuracy localization

- linear: delta_mean `+0.002823`, L1 `0.003674`, delta_tail `+0.001655`, +/-/0 `38/18/0`
  - role `k`: mean `-0.000630`, min `-0.007429`
  - role `q`: mean `+0.000099`, min `-0.001009`
  - role `fc_up`: mean `+0.000538`, min `-0.001592`
  - role_family `qkv`: mean `+0.000628`, min `-0.007429`

## Runtime localization

- `hif4_dynamic_quantize_activation`: 18.389s (100.0%), 56 calls
- `hif4_calibration_and_quantize_weight`: 0.000s (0.0%), 0 calls
- `hif4_calibration_attention`: 0.000s (0.0%), 0 calls
- `hif4_dynamic_quantize_q`: 0.000s (0.0%), 0 calls
- `hif4_dynamic_quantize_k`: 0.000s (0.0%), 0 calls
- `hif4_dynamic_quantize_v`: 0.000s (0.0%), 0 calls
- stages: calibration wall/API `0.000/0.000s`, scoring wall/API `23.087/18.389s`, cache load `1.546s`
- predicted official time: unavailable (requires fresh default panel)

## Next actions

- move work from online Activation quantization into calibration state
- inspect linear role=k first (mean delta=-0.000630, min delta=-0.007429)

> This tool never predicts an official score.
