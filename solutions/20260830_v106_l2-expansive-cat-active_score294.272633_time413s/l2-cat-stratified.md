# L2 stratified Linear candidate screen

> evaluator-side screen; no test output is used to select online state.

- Solution: `D:\工作内容\AI竞赛\solution.py`
- Layers: `[0, 5, 11, 17, 23]`; roles: `q, k, v, o, fc_gate, fc_up, proj`
- Solution LF SHA256: `708081b5281e02da0c2a6e21881027b2e8d31eed423fd3c70e4572424667dd77`
- Elapsed: `75.223s`

| layer | both player | weight perfect | activation perfect |
|---:|---:|---:|---:|
| 0 | 0.61212377 | 0.74994439 | 0.86973172 |
| 5 | 0.48433279 | 0.65235677 | 0.83182894 |
| 11 | 0.52291341 | 0.70979337 | 0.81224927 |
| 17 | 0.50591228 | 0.72212043 | 0.78383676 |
| 23 | 0.50086255 | 0.70781020 | 0.79468999 |

| role | both player | weight perfect | activation perfect |
|---|---:|---:|---:|
| q | 0.65295906 | 0.85721337 | 0.79565847 |
| k | 0.66434283 | 0.90017072 | 0.76796906 |
| v | 0.58427030 | 0.78581581 | 0.79881652 |
| o | 0.52487909 | 0.73215238 | 0.79741085 |
| fc_gate | 0.39727069 | 0.65316702 | 0.74662637 |
| fc_up | 0.43581101 | 0.49233485 | 0.94470375 |
| proj | 0.41706973 | 0.53798107 | 0.87808632 |

Overall selected-layer Linear mean: `0.52522896`.
This is a stratified screen, not the 24-layer parent gate.
