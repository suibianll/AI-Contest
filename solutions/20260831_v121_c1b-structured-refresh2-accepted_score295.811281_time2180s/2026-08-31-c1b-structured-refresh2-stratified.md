# candidate stratified Linear candidate screen

> evaluator-side screen; no test output is used to select online state.

- Solution: `D:\工作内容\AI竞赛\solution.py`
- Layers: `[0, 5, 11, 17, 23]`; roles: `q, k, v, o, fc_gate, fc_up, proj`
- Solution LF SHA256: `1f7633e44538bba6f6bf6be3dd9b2918e6d9a61766c9c3be552268e5b9eafe8c`
- Elapsed: `431.513s`

| layer | both player | weight perfect | activation perfect |
|---:|---:|---:|---:|
| 0 | 0.61833281 | 0.75505531 | 0.86973172 |
| 5 | 0.48734612 | 0.65761870 | 0.82985441 |
| 11 | 0.52889530 | 0.71628244 | 0.81097893 |
| 17 | 0.51166165 | 0.72690022 | 0.78421277 |
| 23 | 0.52074642 | 0.72173275 | 0.79967465 |

| role | both player | weight perfect | activation perfect |
|---|---:|---:|---:|
| q | 0.65651933 | 0.86087589 | 0.79565847 |
| k | 0.67257380 | 0.90746361 | 0.76796906 |
| v | 0.59399813 | 0.79576177 | 0.79881652 |
| o | 0.54815094 | 0.74746701 | 0.80370841 |
| fc_gate | 0.40667706 | 0.66165208 | 0.74617320 |
| fc_up | 0.44227518 | 0.49744213 | 0.94544108 |
| proj | 0.41358077 | 0.53796269 | 0.87446675 |

Overall selected-layer Linear mean: `0.53339646`.
This is a stratified screen, not the 24-layer parent gate.
