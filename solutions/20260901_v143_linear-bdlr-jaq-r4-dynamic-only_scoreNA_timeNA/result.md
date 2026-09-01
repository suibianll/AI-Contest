# v143 result

Status: **REJECTED** (dynamic-only BDLR still regressed Linear).

- Parent: v140 (`52521F1B996BF67641C22A90132ED7A7BCA477976D8A05BEC411CC9E04AA7C90`).
- Change: BDLR rank-4 off-block gradient was enabled only in the final dynamic activation API; calibration/W output selection stayed identical to v140.
- Source SHA256: `DD9DED4FEEFE53B16AC18CB57A4E40D9015D8F616882E64AD904437C84FBAFDA`.
- Protocol: `official-shape-v1`, read-only Qwen2.5-0.5B cache, CUDA, 250 Linear + 200 Attention cases.

| Metric | Value |
|---|---:|
| Linear mean | `0.361153657663258` |
| Attention mean | `0.7159419612310174` |
| Weight calibration | `144.5767321997555 s` |
| Dynamic activation | `20.15450259926729 s` |
| Attention calibration | `37.3762259001378 s` |
| Dynamic Q/K/V | `1.9069569992134348 / 1.830159500706941 / 1.6006686001783237 s` |
| API total | `207.44524579925928 s` |
| Wall | `230.78784280002583 s` |

The time proxy stayed below 300 seconds and Attention was unchanged, but Linear regressed by about `-0.146201` versus v140. The direct rank-4 update is rejected; small-strength probes indicate that a heavily damped update may be useful, so v144 retests the same fixed-cost sketch at strength `0.02`.
