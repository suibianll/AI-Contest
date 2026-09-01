# v144 result

- Parent: v140 (`52521F1B996BF67641C22A90132ED7A7BCA477976D8A05BEC411CC9E04AA7C90`).
- Change: BDLR rank-4 dynamic-only cross-block correction with damping strength `0.02`.
- Source SHA256: `66140D0DA5F1BBCAB7C03968F1756A9E77168B0652F82F4305759BD9AD12344B`.
- Protocol: `official-shape-v1`, read-only Qwen2.5-0.5B cache, CUDA, 250 Linear + 200 Attention cases.

| Metric | Value |
|---|---:|
| Linear mean | `0.5064178886111597` |
| Attention mean | `0.7159419612310174` |
| Weight calibration | `144.7598656003829 s` |
| Dynamic activation | `20.815041100257076 s` |
| Attention calibration | `37.62740510003641 s` |
| Dynamic Q/K/V | `1.8764770993730053 / 1.7730906001525 / 1.5626135003985837 s` |
| API total | `208.4144930006005 s` |
| Wall | `232.17840259999502 s` |

The candidate stayed under the local time proxy but regressed Linear by `-0.0009367485` versus v140; Attention was unchanged. It is rejected. v145 retests the same fixed-cost update at the only smaller strength showing a positive representative-case trend (`0.005`).
