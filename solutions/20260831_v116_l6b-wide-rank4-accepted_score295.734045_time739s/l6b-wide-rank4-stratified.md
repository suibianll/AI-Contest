# candidate stratified Linear candidate screen

> evaluator-side screen; no test output is used to select online state.

- Solution: `D:\工作内容\AI竞赛\solution.py`
- Layers: `[0, 5, 11, 17, 23]`; roles: `q, k, v, o, fc_gate, fc_up, proj`
- Solution LF SHA256: `8fa4db38ac96ca0957e1b1cee61d0c5bd248cf3a4df5d24fa04bedc9239b25f4`
- Elapsed: `130.914s`

| layer | both player | weight perfect | activation perfect |
|---:|---:|---:|---:|
| 0 | 0.61805831 | 0.75479259 | 0.86973172 |
| 5 | 0.48695001 | 0.65726707 | 0.82985441 |
| 11 | 0.52854253 | 0.71600701 | 0.81097893 |
| 17 | 0.51148410 | 0.72671150 | 0.78421277 |
| 23 | 0.52041829 | 0.72128093 | 0.79967465 |

| role | both player | weight perfect | activation perfect |
|---|---:|---:|---:|
| q | 0.65638983 | 0.86073157 | 0.79565847 |
| k | 0.67254360 | 0.90746878 | 0.76796906 |
| v | 0.59377352 | 0.79555669 | 0.79881652 |
| o | 0.54750126 | 0.74684292 | 0.80370841 |
| fc_gate | 0.40656422 | 0.66155795 | 0.74617320 |
| fc_up | 0.44201133 | 0.49717377 | 0.94544108 |
| proj | 0.41285076 | 0.53715105 | 0.87446675 |

Overall selected-layer Linear mean: `0.53309065`.
This is a stratified screen, not the 24-layer parent gate.
