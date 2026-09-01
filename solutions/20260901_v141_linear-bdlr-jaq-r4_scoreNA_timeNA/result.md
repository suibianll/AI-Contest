# v141 result

Status: **REJECTED** (selected-column BDLR caused a large Linear regression).

- Parent: v140 (`52521F1B996BF67641C22A90132ED7A7BCA477976D8A05BEC411CC9E04AA7C90`).
- Change: BDLR-JAQ rank-4 cross-block `H/D` columns were added directly to the activation coordinate gradient.
- Source SHA256: `10521332CC1A616E4495C1D0BD5527E011C91A062D46D9E44644D3FB789B1BED`.
- Protocol: `official-shape-v1`, read-only Qwen2.5-0.5B cache, CUDA, 250 Linear + 200 Attention cases.

| Metric | Value |
|---|---:|
| Linear mean | `0.2817598810465049` |
| Attention mean | `0.7159419612310174` |
| Weight calibration | `145.21477789839264 s` |
| Dynamic activation | `16.653756899642758 s` |
| Attention calibration | `37.62567089963704 s` |
| Dynamic Q/K/V | `1.8808799997204915 / 1.750646100495942 / 1.5553506996948272 s` |
| API total | `204.6810824975837 s` |
| Wall | `228.12701109994669 s` |

The candidate stayed within the local time proxy but Linear regressed from v140 by about `-0.225595`; Attention was unchanged. The direct rank-4 gradient is therefore rejected. The failure analysis found that anchor coordinates were updated with an asymmetric column-only sketch; v142 keeps the same fixed-cost sketch but freezes anchor codes before retesting. Official score/time remain unregistered.
