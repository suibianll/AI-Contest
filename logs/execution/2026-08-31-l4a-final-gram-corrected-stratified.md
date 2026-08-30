# candidate stratified Linear candidate screen

> evaluator-side screen; no test output is used to select online state.

- Solution: `D:\工作内容\AI竞赛\solution.py`
- Layers: `[0, 5, 11, 17, 23]`; roles: `q, k, v, o, fc_gate, fc_up, proj`
- Solution LF SHA256: `8e78bf5d94c1f47d6281a673892ca0f2a06004e735976c0854bf9386eb498ebc`
- Elapsed: `80.393s`

| layer | both player | weight perfect | activation perfect |
|---:|---:|---:|---:|
| 0 | 0.61626089 | 0.75312569 | 0.86973172 |
| 5 | 0.48843724 | 0.65643568 | 0.83182894 |
| 11 | 0.52604813 | 0.71306542 | 0.81224927 |
| 17 | 0.50946059 | 0.72545502 | 0.78383676 |
| 23 | 0.50451823 | 0.71141632 | 0.79468999 |

| role | both player | weight perfect | activation perfect |
|---|---:|---:|---:|
| q | 0.65573462 | 0.86005113 | 0.79565847 |
| k | 0.67062548 | 0.90586272 | 0.76796906 |
| v | 0.59124290 | 0.79297084 | 0.79881652 |
| o | 0.53102352 | 0.73739712 | 0.79741085 |
| fc_gate | 0.40013426 | 0.65581156 | 0.74662637 |
| fc_up | 0.43678461 | 0.49322294 | 0.94470375 |
| proj | 0.41706973 | 0.53798107 | 0.87808632 |

Overall selected-layer Linear mean: `0.52894502`.
This is a stratified screen, not the 24-layer parent gate.
