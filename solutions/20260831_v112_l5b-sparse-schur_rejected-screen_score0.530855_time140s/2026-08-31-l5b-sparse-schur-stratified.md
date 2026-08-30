# candidate stratified Linear candidate screen

> evaluator-side screen; no test output is used to select online state.

- Solution: `D:\工作内容\AI竞赛\solution.py`
- Layers: `[0, 5, 11, 17, 23]`; roles: `q, k, v, o, fc_gate, fc_up, proj`
- Solution LF SHA256: `94a06fcce29b3e6639c4dab4d8c96e4e37f4f74947adec6e1f57b87512e0bc9e`
- Elapsed: `140.300s`

| layer | both player | weight perfect | activation perfect |
|---:|---:|---:|---:|
| 0 | 0.61670768 | 0.75352817 | 0.86973172 |
| 5 | 0.48117237 | 0.65551455 | 0.82528039 |
| 11 | 0.53441788 | 0.71506577 | 0.81772538 |
| 17 | 0.50657722 | 0.72574640 | 0.78019240 |
| 23 | 0.51540036 | 0.72032173 | 0.79594314 |

| role | both player | weight perfect | activation perfect |
|---|---:|---:|---:|
| q | 0.65479135 | 0.86003931 | 0.79500627 |
| k | 0.67062548 | 0.90586272 | 0.76796906 |
| v | 0.59340994 | 0.79272437 | 0.80034227 |
| o | 0.53794142 | 0.74596621 | 0.79502361 |
| fc_gate | 0.40618361 | 0.66113440 | 0.74617320 |
| fc_up | 0.44192540 | 0.49708107 | 0.94544108 |
| proj | 0.41110851 | 0.53543918 | 0.87446675 |

Overall selected-layer Linear mean: `0.53085510`.
This is a stratified screen, not the 24-layer parent gate.
