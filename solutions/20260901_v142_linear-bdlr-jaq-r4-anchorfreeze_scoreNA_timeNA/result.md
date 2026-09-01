# v142 result

- Parent: v141 (`10521332CC1A616E4495C1D0BD5527E011C91A062D46D9E44644D3FB789B1BED`).
- Change: freeze the four selected BDLR anchor codes while applying the rank-4 off-block gradient to the other coordinates.
- Source SHA256: `05EA39634891430EF24DD879F982355AC7275534D94C4EBC325646F1C5BB1347`.
- Protocol: `official-shape-v1`, read-only Qwen2.5-0.5B cache, CUDA, 250 Linear + 200 Attention cases.

| Metric | Value |
|---|---:|
| Linear mean | `0.2825593037789877` |
| Attention mean | `0.7159419612310174` |
| Weight calibration | `148.65847110084724 s` |
| Dynamic activation | `20.038276001112536 s` |
| Attention calibration | `37.55544249981176 s` |
| Dynamic Q/K/V | `1.8702348999213427 / 1.7656643001828343 / 1.5719974000239745 s` |
| API total | `211.46008620189968 s` |
| Wall | `234.84197629999835 s` |

Freezing anchors recovered only `0.0007994` Linear over v141; Linear remained far below v140 while Attention stayed unchanged. The candidate is rejected. The next v143 isolates BDLR to the online activation pass so it cannot perturb the established output-supervised weight calibration.
