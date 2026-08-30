# candidate stratified Linear candidate screen

> evaluator-side screen; no test output is used to select online state.

- Solution: `D:\工作内容\AI竞赛\solution.py`
- Layers: `[0, 5, 11, 17, 23]`; roles: `q, k, v, o, fc_gate, fc_up, proj`
- Solution LF SHA256: `043e5401c7d8cf68339e9faec3f60943c11821e3b51bb1563d2ecd8a812f22e5`
- Elapsed: `128.876s`

| layer | both player | weight perfect | activation perfect |
|---:|---:|---:|---:|
| 0 | 0.61789733 | 0.75462635 | 0.86973172 |
| 5 | 0.48673474 | 0.65700441 | 0.82985441 |
| 11 | 0.52819934 | 0.71566729 | 0.81097893 |
| 17 | 0.51126569 | 0.72650380 | 0.78421277 |
| 23 | 0.52011166 | 0.72103449 | 0.79967465 |

| role | both player | weight perfect | activation perfect |
|---|---:|---:|---:|
| q | 0.65638983 | 0.86073157 | 0.79565847 |
| k | 0.67254360 | 0.90746878 | 0.76796906 |
| v | 0.59377352 | 0.79555669 | 0.79881652 |
| o | 0.54750126 | 0.74684292 | 0.80370841 |
| fc_gate | 0.40656422 | 0.66155795 | 0.74617320 |
| fc_up | 0.44201133 | 0.49717377 | 0.94544108 |
| proj | 0.41110851 | 0.53543918 | 0.87446675 |

Overall selected-layer Linear mean: `0.53284175`.
This is a stratified screen, not the 24-layer parent gate.
