# candidate stratified Linear candidate screen

> evaluator-side screen; no test output is used to select online state.

- Solution: `D:\工作内容\AI竞赛\solution.py`
- Layers: `[0, 5, 11, 17, 23]`; roles: `q, k, v, o, fc_gate, fc_up, proj`
- Solution LF SHA256: `b498a6d5cf025b58807c9751d0e98e237f0ea9d5bdf2a11fc5be8f2aa7ae1801`
- Elapsed: `429.949s`

| layer | both player | weight perfect | activation perfect |
|---:|---:|---:|---:|
| 0 | 0.61828335 | 0.75501143 | 0.86973172 |
| 5 | 0.48729740 | 0.65756437 | 0.82985441 |
| 11 | 0.52885671 | 0.71624420 | 0.81097893 |
| 17 | 0.51163112 | 0.72686492 | 0.78421277 |
| 23 | 0.52069000 | 0.72170767 | 0.79967465 |

| role | both player | weight perfect | activation perfect |
|---|---:|---:|---:|
| q | 0.65651933 | 0.86087589 | 0.79565847 |
| k | 0.67257380 | 0.90746361 | 0.76796906 |
| v | 0.59399813 | 0.79576177 | 0.79881652 |
| o | 0.54815094 | 0.74746701 | 0.80370841 |
| fc_gate | 0.40667706 | 0.66165208 | 0.74617320 |
| fc_up | 0.44227518 | 0.49744213 | 0.94544108 |
| proj | 0.41326755 | 0.53768713 | 0.87446675 |

Overall selected-layer Linear mean: `0.53335171`.
This is a stratified screen, not the 24-layer parent gate.
