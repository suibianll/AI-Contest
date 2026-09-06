# proxy-v3 diagnosis

**Decision: `continue_next_shard`**

## Warnings

- no official-time prediction: run a fresh default panel without calibration cache

## Accuracy localization

- linear: delta_mean `+0.001251`, L1 `0.002190`, delta_tail `+0.001049`, +/-/0 `37/19/0`
  - role `k`: mean `-0.001006`, min `-0.005000`
  - role_family `o`: mean `-0.000053`, min `-0.001358`
  - role `o`: mean `-0.000053`, min `-0.001358`
  - shape_bucket `hidden_to_hidden`: mean `-0.000043`, min `-0.001856`

## Runtime localization

- `hif4_dynamic_quantize_activation`: 18.673s (100.0%), 56 calls
- `hif4_calibration_and_quantize_weight`: 0.000s (0.0%), 0 calls
- `hif4_calibration_attention`: 0.000s (0.0%), 0 calls
- `hif4_dynamic_quantize_q`: 0.000s (0.0%), 0 calls
- `hif4_dynamic_quantize_k`: 0.000s (0.0%), 0 calls
- `hif4_dynamic_quantize_v`: 0.000s (0.0%), 0 calls
- stages: calibration wall/API `0.000/0.000s`, scoring wall/API `23.453/18.673s`, cache load `1.506s`
- predicted official time: unavailable (requires fresh default panel)

## Next actions

- move work from online Activation quantization into calibration state
- inspect linear role=k first (mean delta=-0.001006, min delta=-0.005000)

> This tool never predicts an official score.
