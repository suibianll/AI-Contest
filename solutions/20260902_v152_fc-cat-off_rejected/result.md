# v152 — fc CAT-off role control

## Status

`REJECTED` — the Qwen paired panel shows only a small mixed-sign fc gain; this is not a stable
candidate for promotion.

## Parent and change

- Parent: pre-A3 v147 fixed combination (v140 Linear + v86 Attention), workbench SHA
  `800CA10EC3414E4FE886B93CA62BD4A350D26BBA015287DF7E8DF2DD871AC23D`.
- Single mechanism: skip the expansive (`rows > channels`) CAT balance while retaining BOAT and all
  other Linear/Attention code.
- This is a role control motivated by the external hif4 fc attribution; it does not change the root
  source or the Attention path.

## Canonical Qwen proxy-v2 evidence

The same read-only cache and full calibration call graph were used for both sides. The 14-case panel
covers two layers × seven static roles; the 56-case panel covers the same seven roles across the
stratified eight-layer subset. Attention uses one frozen v86 case.

| Panel | Candidate | Linear | Attention | Overall | API total | Wall |
|---|---|---:|---:|---:|---:|---:|
| 14 Linear + 1 Attention | pre-A3 parent | 0.582528216 | 0.942927486 | 0.606554834 | 201.258s | 209.078s |
| 14 Linear + 1 Attention | v152 | 0.583139209 | 0.942927486 | 0.607125094 | 199.578s | 206.988s |
| 56 Linear + 1 Attention | pre-A3 parent | 0.542366307 | 0.942927486 | 0.549393696 | 201.120s | 212.275s |
| 56 Linear + 1 Attention | v152 | 0.542552798 | 0.942927486 | 0.549576915 | 200.432s | 213.442s |

On the 14-case panel, fc_gate improved `0.396959→0.403497` while fc_up fell
`0.368327→0.366066`. On the 56-case paired panel, fc_gate improved `+0.001871` in mean gain,
fc_up fell `−0.000565`, and the family delta was only `+0.000653` with mixed signs (3 positive,
3 negative fc layers in the selected panel). q/k/v/o/proj and Attention were bit-identical. The
sub-second API difference is measurement noise, not a runtime claim.

## External cross-check

The upstream hif4 four-layer GPT-2 causal smoke (`amax6`, `seq=128`, `calib=2`, `test=2`) changed
only fc, from `.5658` to `.5709`; proj and Attention stayed unchanged. This corroborates the role
direction but not its Qwen transfer or official score.

## Official result

- Score: `unregistered`.
- Time: `unregistered`.
- Status: `unregistered`.

## Decision

Do not change root `solution.py` and do not use v152 as a parent. Keep the result as evidence that
removing CAT may be directionally useful for fc, but the effect is too small and layer-mixed. Proceed
to the L1 decoupled encoder with a fixed-code closed-form E6M2 stored-scale update; retain BOAT.

## Source

- SHA256: `1CA4A1B0A428A17D5EB9F66FC1CD6FFA3300550CA3001BD7C27AE129D3662D69`
- Source: `solution.py`
