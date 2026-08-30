# candidate stratified Linear candidate screen

> evaluator-side screen; no test output is used to select online state.

- Solution: `D:\工作内容\AI竞赛\solution.py`
- Layers: `[0, 5, 11, 17, 23]`; roles: `q, k, v, o, fc_gate, fc_up, proj`
- Solution LF SHA256: `c9c45a7911594b4b378d0c5e2769187d76dc587d79b6da9fa5f5a487e4b7cb11`
- Elapsed: `403.523s`

| layer | both player | weight perfect | activation perfect |
|---:|---:|---:|---:|
| 0 | 0.61831709 | 0.75504800 | 0.86973172 |
| 5 | 0.48732706 | 0.65760193 | 0.82985441 |
| 11 | 0.52886733 | 0.71625758 | 0.81097893 |
| 17 | 0.51166348 | 0.72690196 | 0.78421277 |
| 23 | 0.52070163 | 0.72171318 | 0.79967465 |

| role | both player | weight perfect | activation perfect |
|---|---:|---:|---:|
| q | 0.65651933 | 0.86087589 | 0.79565847 |
| k | 0.67257380 | 0.90746361 | 0.76796906 |
| v | 0.59399813 | 0.79576177 | 0.79881652 |
| o | 0.54815094 | 0.74746701 | 0.80370841 |
| fc_gate | 0.40667706 | 0.66165208 | 0.74617320 |
| fc_up | 0.44227518 | 0.49744213 | 0.94544108 |
| proj | 0.41343278 | 0.53786924 | 0.87446675 |

Overall selected-layer Linear mean: `0.53337532`.
This is a stratified screen, not the 24-layer parent gate.
