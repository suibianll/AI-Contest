# v145 result

Status: **REJECTED** (damping `0.005` remained below v140).

- Parent: v140 (`52521F1B996BF67641C22A90132ED7A7BCA477976D8A05BEC411CC9E04AA7C90`).
- Change: dynamic-only BDLR rank-4 cross-block column correction with damping strength `0.005`.
- Source SHA256: `E028E360A845867917669914FA9CC3CFF12F640B2923E0B8A8B67685EBF0052B`.
- Protocol: `official-shape-v1`, read-only Qwen2.5-0.5B cache, CUDA, 250 Linear + 200 Attention cases.

| Metric | Value |
|---|---:|
| Linear mean | `0.5062555833608251` |
| Attention mean | `0.7159419612310174` |
| Weight calibration | `145.61814730090555 s` |
| Dynamic activation | `20.392538601648994 s` |
| Attention calibration | `37.23012850002851 s` |
| Dynamic Q/K/V | `1.9061777002643794 / 1.7881297999992967 / 1.5776650006882846 s` |
| API total | `208.51278690353502 s` |
| Wall | `232.20573719998356 s` |

The candidate remained under the local time proxy but regressed Linear by `-0.0010990538` versus v140; Attention was unchanged. Together with v141–v144, this rejects the selected-column BDLR approximation even after isolating it to dynamic A and damping it. The active root is restored to v140; the next experiment is a symmetric/HOQ joint code-domain update.
