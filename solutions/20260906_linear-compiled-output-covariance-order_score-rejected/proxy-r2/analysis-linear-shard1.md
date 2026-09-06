# proxy-v3 diagnosis

**Decision: `continue_next_shard`**

## Warnings

- no official-time prediction: run a fresh default panel without calibration cache

## Accuracy localization

- linear: delta_mean `+0.000278`, L1 `0.001062`, delta_tail `+0.000289`, +/-/0 `30/18/8`
  - layer `1`: mean `-0.000277`, min `-0.004702`
  - split `validation`: mean `-0.000183`, min `-0.004702`
  - role_family `proj`: mean `+0.000000`, min `+0.000000`
  - role `proj`: mean `+0.000000`, min `+0.000000`

## Runtime localization

- `hif4_dynamic_quantize_activation`: 17.027s (100.0%), 56 calls
- `hif4_calibration_and_quantize_weight`: 0.000s (0.0%), 0 calls
- `hif4_calibration_attention`: 0.000s (0.0%), 0 calls
- `hif4_dynamic_quantize_q`: 0.000s (0.0%), 0 calls
- `hif4_dynamic_quantize_k`: 0.000s (0.0%), 0 calls
- `hif4_dynamic_quantize_v`: 0.000s (0.0%), 0 calls
- stages: calibration wall/API `0.000/0.000s`, scoring wall/API `20.889/17.027s`, cache load `1.415s`
- predicted official time: unavailable (requires fresh default panel)

## Next actions

- move work from online Activation quantization into calibration state
- inspect linear layer=1 first (mean delta=-0.000277, min delta=-0.004702)

> This tool never predicts an official score.
