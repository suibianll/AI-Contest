# v140 result

Status: **REJECTED / OFFICIAL PASS BUT INFERIOR**. The user-confirmed official result is
`15838 / 207s`; this is below the v86 baseline (`16744 / 222.7s`) and far below the current
17816 anchor, so the v138–v145 lineage remains closed. The local Linear gain over v138 was only
`0.00003513`.

- Parent: v138 time-safe root (`3A120BEB62443FF6A5BCDB89B5FAD970AC6D8D45F48F40FE31812073060C2D10`).
- Change: ROAB-P2 reciprocal 2x2 pair transforms are learned from calibration activation/weight covariance, applied as an exact (within float32) reciprocal coordinate change to `A` and `W`, and selected by the legal `Q(A)Q(W)^T` output score. Attention remains the v138 static path.
- Protocol: `official-shape-v1`, Qwen2.5-0.5B cache, 250 Linear + 200 Attention cases, CUDA, read-only cache.
- Source SHA256: `52521F1B996BF67641C22A90132ED7A7BCA477976D8A05BEC411CC9E04AA7C90`.

| Metric | Value |
|---|---:|
| Linear mean | `0.5073546371426622` |
| Attention mean | `0.7159419612310174` |
| Weight calibration | `146.88725969940424 s` |
| Dynamic activation | `16.15407369856257 s` |
| Attention calibration | `37.15961650002282 s` |
| Dynamic Q/K/V | `1.869824199588038 / 1.751266300212592 / 1.543029900756665 s` |
| API total | `205.36507029854693 s` |
| Wall | `229.33724829996936 s` |

Compared with v138, Linear improves by `+0.0000351322807745`; Attention is unchanged bit-for-bit.
The official result is recorded above, but the candidate is rejected because it loses `−1029` points
to v86 and `−1978` points to the 17816 anchor. It is not the active root or a future parent.

Exact JSON/report: [`v140 JSON`](../../artifacts/official_eval/v140-linear-roab-pair-official-shape-v1.json), [`v140 report`](../../logs/official_eval/v140-linear-roab-pair-official-shape-v1.md).
