# v153 — fc decoupled Activation scale (first implementation)

## Status

`REJECTED` — direct high-precision `s_q` code assignment without a fitted stored scale causes a
large fc regression on the canonical Qwen panel.

## Parent and change

- Parent: pre-A3 v147 fixed combination, SHA
  `800CA10EC3414E4FE886B93CA62BD4A350D26BBA015287DF7E8DF2DD871AC23D`.
- Single mechanism: for expansive fc shapes only, Activation encoding uses the BF16 pre-E6M2
  `a_max/7` as `s_q` for lv2/lv3/mantissa assignment; the stored scale remains the legal E6M2
  candidate `s_d`. Weight, BOAT/ROAB, output refinement, and v86 Attention are frozen.
- The deployment path adds no API calls or Python candidate loop and passes independent HiF4 state
  validation on a synthetic round-trip.

## Canonical Qwen proxy-v2 targeted evidence

Protocol: existing `qwen2.5-0.5b-proxy-v2.pt`, 14 Linear cases (two layers × seven roles) + 1
Attention case, shared 168 Weight + 24 Attention calibration states.

| Candidate | Linear | Attention | Overall | API total | Wall |
|---|---:|---:|---:|---:|---:|
| pre-A3 parent | 0.582528216 | 0.942927486 | 0.606554834 | 201.258s | 209.078s |
| v153 | 0.568753650 | 0.942927486 | 0.593698573 | 197.656s | 205.470s |

Static roles changed only in fc: fc_gate `0.396959→0.350049`, fc_up `0.368327→0.318816`, and
fc family `0.382643→0.334432`. q/k/v/o/proj and Attention were identical. The lower wall time is
not an accepted performance result because the candidate is inaccurate.

## Diagnosis and next step

The failure is consistent with the scale mismatch in the implementation: code assignment moved to
`s_q`, but `s_d` was not re-estimated after the code was fixed. The next implementation must first
fit `s_d` in the deployed Weight/Gram metric and then project to legal E6M2, as specified by L1; do
not widen `s_q` into a sweep.

## Official result

- Score: `unregistered`.
- Time: `unregistered`.
- Status: `unregistered`.

## Source

- SHA256: `641A0FEB6B113131F215ED855F4799F99632DF5F22D8A10C83915056836E908E`
- Source: `solution.py`
