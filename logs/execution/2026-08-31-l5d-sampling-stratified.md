# candidate stratified Linear candidate screen

> evaluator-side screen; no test output is used to select online state.

- Solution: `D:\工作内容\AI竞赛\solution.py`
- Layers: `[0, 5, 11, 17, 23]`; roles: `q, k, v, o, fc_gate, fc_up, proj`
- Solution LF SHA256: `4b6fa66827eabaeebbdf5289ba0c936aab43e610b36981527c402b76a2d617d5`
- Elapsed: `125.332s`

| layer | both player | weight perfect | activation perfect |
|---:|---:|---:|---:|
| 0 | 0.61670768 | 0.75352817 | 0.86973172 |
| 5 | 0.49017517 | 0.65771735 | 0.83259417 |
| 11 | 0.51339393 | 0.70665325 | 0.80472355 |
| 17 | 0.50973818 | 0.72573327 | 0.78383676 |
| 23 | 0.50654254 | 0.72032092 | 0.78762148 |

| role | both player | weight perfect | activation perfect |
|---|---:|---:|---:|
| q | 0.64788800 | 0.86220362 | 0.78576189 |
| k | 0.67062548 | 0.90586272 | 0.76796906 |
| v | 0.59124290 | 0.79297084 | 0.79881652 |
| o | 0.51773002 | 0.73248962 | 0.78854798 |
| fc_gate | 0.40629167 | 0.66141320 | 0.74600055 |
| fc_up | 0.44033270 | 0.49661307 | 0.94472844 |
| proj | 0.41706973 | 0.53798107 | 0.87808632 |

Overall selected-layer Linear mean: `0.52731150`.
This is a stratified screen, not the 24-layer parent gate.
