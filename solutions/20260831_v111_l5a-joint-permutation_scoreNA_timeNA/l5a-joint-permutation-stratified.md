# candidate stratified Linear candidate screen

> evaluator-side screen; no test output is used to select online state.

- Solution: `D:\工作内容\AI竞赛\solution.py`
- Layers: `[0, 5, 11, 17, 23]`; roles: `q, k, v, o, fc_gate, fc_up, proj`
- Solution LF SHA256: `6b229081121c4a7edd69575c93dc01488be8f8b5e1479007522421e93e1adc57`
- Elapsed: `136.616s`

| layer | both player | weight perfect | activation perfect |
|---:|---:|---:|---:|
| 0 | 0.61670768 | 0.75352817 | 0.86973172 |
| 5 | 0.48554060 | 0.65570158 | 0.82985441 |
| 11 | 0.52760337 | 0.71505739 | 0.81097893 |
| 17 | 0.51050457 | 0.72576016 | 0.78421277 |
| 23 | 0.51907851 | 0.72031000 | 0.79967465 |

| role | both player | weight perfect | activation perfect |
|---|---:|---:|---:|
| q | 0.65573462 | 0.86005113 | 0.79565847 |
| k | 0.67062548 | 0.90586272 | 0.76796906 |
| v | 0.59124290 | 0.79297084 | 0.79881652 |
| o | 0.54638810 | 0.74596088 | 0.80370841 |
| fc_gate | 0.40618361 | 0.66113440 | 0.74617320 |
| fc_up | 0.44192540 | 0.49708107 | 0.94544108 |
| proj | 0.41110851 | 0.53543918 | 0.87446675 |

Overall selected-layer Linear mean: `0.53188695`.
This is a stratified screen, not the 24-layer parent gate.
