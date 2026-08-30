# L3b1 stratified Linear candidate screen

> evaluator-side screen; no test output is used to select online state.

- Solution: `D:\工作内容\AI竞赛\solution.py`
- Layers: `[0, 5, 11, 17, 23]`; roles: `q, k, v, o, fc_gate, fc_up, proj`
- Solution LF SHA256: `6ca6d86a47bc437a8ad8bcf0f1347f8c4552c34c1ad440f674e0498d5b6add3c`
- Elapsed: `74.801s`

| layer | both player | weight perfect | activation perfect |
|---:|---:|---:|---:|
| 0 | 0.61312291 | 0.75064230 | 0.86973172 |
| 5 | 0.48547337 | 0.65353524 | 0.83182894 |
| 11 | 0.52362976 | 0.71048459 | 0.81224927 |
| 17 | 0.50671749 | 0.72294175 | 0.78383676 |
| 23 | 0.50155918 | 0.70848747 | 0.79468999 |

| role | both player | weight perfect | activation perfect |
|---|---:|---:|---:|
| q | 0.65361862 | 0.85784825 | 0.79565847 |
| k | 0.66523762 | 0.90119968 | 0.76796906 |
| v | 0.58625808 | 0.78779538 | 0.79881652 |
| o | 0.52652732 | 0.73337742 | 0.79741085 |
| fc_gate | 0.39785429 | 0.65368574 | 0.74662637 |
| fc_up | 0.43613811 | 0.49264036 | 0.94470375 |
| proj | 0.41706973 | 0.53798107 | 0.87808632 |

Overall selected-layer Linear mean: `0.52610054`.
This is a stratified screen, not the 24-layer parent gate.
