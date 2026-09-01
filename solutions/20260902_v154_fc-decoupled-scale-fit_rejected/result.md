# v154 — fc decoupled scale fit

## Status

`REJECTED` — the fixed-code stored-scale fit is legal and bounded, but it does not recover the
v153 regression on the canonical panel; it produces the same scored fc output as v153.

## Parent and change

- Parent: v153 direct fc decoupled Activation encoder, whose base parent is pre-A3 v147; pre-A3
  SHA `800CA10EC3414E4FE886B93CA62BD4A350D26BBA015287DF7E8DF2DD871AC23D`.
- Single mechanism: after `s_q` fixes lv2/lv3/mantissa code assignment, fit one scalar `s_d` per
  64-channel block in the deployed `Q(W)` Gram metric and project it once to nearest legal E6M2.
  No new candidate loop, API call, transform, or Attention change.

## Canonical Qwen proxy-v2 targeted evidence

Protocol: existing `qwen2.5-0.5b-proxy-v2.pt`, 14 Linear cases (two layers × seven roles) + 1
Attention case; shared 168 Weight + 24 Attention calibration states.

| Candidate | Linear | Attention | Overall | API total | Wall |
|---|---:|---:|---:|---:|---:|
| pre-A3 parent | 0.582528216 | 0.942927486 | 0.606554834 | 201.258s | 209.078s |
| v153 | 0.568753650 | 0.942927486 | 0.593698573 | 197.656s | 205.470s |
| v154 | 0.568753650 | 0.942927486 | 0.593698573 | 198.098s | 205.510s |

v154 and v153 have identical role means: fc_gate `0.350049`, fc_up `0.318816`, fc family
`0.334432`; q/k/v/o/proj and Attention remain unchanged. The scale fitting is therefore not yet
an effective encoder improvement for this panel, even though synthetic round-trip validation and
the reference legal-state check pass.

## Decision

Do not change root `solution.py` and do not use v154 as a parent. Stop this direct decoupled-scale
variant until a teacher/oracle shows code assignments with recoverable margin; move the next work to
the plan's bounded L3 oracle-to-encoder diagnostic rather than another scale or CAT parameter tweak.

## Official result

- Score: `unregistered`.
- Time: `unregistered`.
- Status: `unregistered`.

## Source

- SHA256: `631956816BFFD94D1EDF321B581AFE706327621966EEACE10B9A87A7628CBF86`
- Source: `solution.py`
