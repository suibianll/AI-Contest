# L4a stratified Linear candidate screen

> evaluator-side screen; no test output is used to select online state.

- Solution: `D:\工作内容\AI竞赛\solution.py`
- Layers: `[0, 5, 11, 17, 23]`; roles: `q, k, v, o, fc_gate, fc_up, proj`
- Solution LF SHA256: `c71078b773612541a94ed47f146b0f26349c10ca13ce324ac7fdcc9ea67e2bf8`
- Elapsed: `78.281s`

| layer | both player | weight perfect | activation perfect |
|---:|---:|---:|---:|
| 0 | 0.61624446 | 0.75309867 | 0.86973172 |
| 5 | 0.48844566 | 0.65645348 | 0.83182894 |
| 11 | 0.52607451 | 0.71298953 | 0.81224927 |
| 17 | 0.50950236 | 0.72549726 | 0.78383676 |
| 23 | 0.50447956 | 0.71139296 | 0.79468999 |

| role | both player | weight perfect | activation perfect |
|---|---:|---:|---:|
| q | 0.65573462 | 0.86005113 | 0.79565847 |
| k | 0.67062548 | 0.90586272 | 0.76796906 |
| v | 0.59124290 | 0.79297084 | 0.79881652 |
| o | 0.53102352 | 0.73739712 | 0.79741085 |
| fc_gate | 0.40007904 | 0.65565706 | 0.74662637 |
| fc_up | 0.43686986 | 0.49328473 | 0.94470375 |
| proj | 0.41706973 | 0.53798107 | 0.87808632 |

Overall selected-layer Linear mean: `0.52894931`.
This is a stratified screen, not the 24-layer parent gate.
