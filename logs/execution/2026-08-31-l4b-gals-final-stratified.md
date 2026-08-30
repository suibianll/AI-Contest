# candidate stratified Linear candidate screen

> evaluator-side screen; no test output is used to select online state.

- Solution: `D:\工作内容\AI竞赛\solution.py`
- Layers: `[0, 5, 11, 17, 23]`; roles: `q, k, v, o, fc_gate, fc_up, proj`
- Solution LF SHA256: `1cf3e1e8ad436633b528c281bcda506b3e101e433f582b1c874019a5280b17eb`
- Elapsed: `125.490s`

| layer | both player | weight perfect | activation perfect |
|---:|---:|---:|---:|
| 0 | 0.61632877 | 0.75314874 | 0.86973172 |
| 5 | 0.48844515 | 0.65644040 | 0.83182894 |
| 11 | 0.52602545 | 0.71303987 | 0.81224927 |
| 17 | 0.50943194 | 0.72543402 | 0.78383676 |
| 23 | 0.50449081 | 0.71138811 | 0.79468999 |

| role | both player | weight perfect | activation perfect |
|---|---:|---:|---:|
| q | 0.65573462 | 0.86005113 | 0.79565847 |
| k | 0.67062548 | 0.90586272 | 0.76796906 |
| v | 0.59124290 | 0.79297084 | 0.79881652 |
| o | 0.53102352 | 0.73739712 | 0.79741085 |
| fc_gate | 0.40017324 | 0.65578141 | 0.74662637 |
| fc_up | 0.43674149 | 0.49318731 | 0.94470375 |
| proj | 0.41706973 | 0.53798107 | 0.87808632 |

Overall selected-layer Linear mean: `0.52894443`.
This is a stratified screen, not the 24-layer parent gate.
