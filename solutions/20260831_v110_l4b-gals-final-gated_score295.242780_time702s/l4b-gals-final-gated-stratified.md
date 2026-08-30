# candidate stratified Linear candidate screen

> evaluator-side screen; no test output is used to select online state.

- Solution: `D:\工作内容\AI竞赛\solution.py`
- Layers: `[0, 5, 11, 17, 23]`; roles: `q, k, v, o, fc_gate, fc_up, proj`
- Solution LF SHA256: `b6d69e17c1224bfaba1c28bf22ee71b63da0ad388a22d0fbbb4c016b290953af`
- Elapsed: `125.607s`

| layer | both player | weight perfect | activation perfect |
|---:|---:|---:|---:|
| 0 | 0.61670768 | 0.75352817 | 0.86973172 |
| 5 | 0.48868371 | 0.65668915 | 0.83182894 |
| 11 | 0.52643546 | 0.71344421 | 0.81224927 |
| 17 | 0.50973818 | 0.72573327 | 0.78383676 |
| 23 | 0.50489544 | 0.71175821 | 0.79468999 |

| role | both player | weight perfect | activation perfect |
|---|---:|---:|---:|
| q | 0.65573462 | 0.86005113 | 0.79565847 |
| k | 0.67062548 | 0.90586272 | 0.76796906 |
| v | 0.59124290 | 0.79297084 | 0.79881652 |
| o | 0.53102352 | 0.73739712 | 0.79741085 |
| fc_gate | 0.40164746 | 0.65723135 | 0.74662637 |
| fc_up | 0.43770093 | 0.49411998 | 0.94470375 |
| proj | 0.41706973 | 0.53798107 | 0.87808632 |

Overall selected-layer Linear mean: `0.52929209`.
This is a stratified screen, not the 24-layer parent gate.
