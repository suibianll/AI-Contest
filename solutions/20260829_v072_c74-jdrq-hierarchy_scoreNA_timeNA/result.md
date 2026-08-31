# C74 JDRQ fixed-Q(A) hierarchy residual

- Parent: current C69-series root
- Unique mechanism: after activation state freeze, use calibration `Z=Q(A)` and
  transformed teacher products only to search legal E6M2/lv2/lv3/mantissa
  updates for static Q(W); no product tensor enters `activation_state`.
- Root/archive SHA256: `61C216BEE1ECA9DB6185BCD49C679A34B684F540451CD5562A2008D0DA4B2AD9`
- Official result (user-confirmed 2026-08-31): **`22662 / 226s`**, successful;
  Attention passed. The uploaded-package SHA was not reported separately.
- Full dual/block ridge target is implemented but remains disabled by default
  until its calibration-to-hidden migration gap is reduced.

## Local screen

| model | native total | API time |
|---|---:|---:|
| GPT-2 small | 160.571830 | 61.88s (CUDA) |
| Qwen2.5-0.5B | 356.605602 | 163.41s (CUDA) |
| OPT-125M | 85.580941 | 59.56s (CUDA) |
| Pythia-160M | 179.059425 | 59.71s (CUDA) |

All reported API times are below the 420-second limit. Focused JDRQ,
reference-codec, linear-compliance, and release tests pass; see
`logs/execution/2026-08-29-jdrq-execution.md`.
