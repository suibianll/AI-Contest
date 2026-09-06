# proxy-v3 diagnosis

**Decision: `continue_next_shard`**

## Warnings

- no official-time prediction: run a fresh default panel without calibration cache

## Accuracy localization

- linear: delta_mean `+0.001907`, L1 `0.002376`, delta_tail `+0.001148`, +/-/0 `44/12/0`
  - role `q`: mean `+0.000110`, min `-0.002367`
  - layer `0`: mean `+0.000767`, min `-0.003074`
  - shape_bucket `hidden_to_hidden`: mean `+0.000852`, min `-0.002367`
  - role `fc_up`: mean `+0.000861`, min `-0.000421`

## Runtime localization

- `hif4_dynamic_quantize_activation`: 16.996s (100.0%), 56 calls
- `hif4_calibration_and_quantize_weight`: 0.000s (0.0%), 0 calls
- `hif4_calibration_attention`: 0.000s (0.0%), 0 calls
- `hif4_dynamic_quantize_q`: 0.000s (0.0%), 0 calls
- `hif4_dynamic_quantize_k`: 0.000s (0.0%), 0 calls
- `hif4_dynamic_quantize_v`: 0.000s (0.0%), 0 calls
- stages: calibration wall/API `0.000/0.000s`, scoring wall/API `20.878/16.996s`, cache load `1.568s`
- predicted official time: unavailable (requires fresh default panel)

## Next actions

- move work from online Activation quantization into calibration state
- inspect linear role=q first (mean delta=+0.000110, min delta=-0.002367)

> This tool never predicts an official score.
