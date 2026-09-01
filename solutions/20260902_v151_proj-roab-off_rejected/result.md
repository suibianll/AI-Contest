# v151 — proj ROAB-off role control

## Status

`REJECTED` — local Qwen panel no-op; external GPT-2-only improvement is insufficient for promotion.

## Parent and change

- Parent: pre-A3 v147 fixed combination (v140 Linear + v86 Attention), workbench SHA
  `800CA10EC3414E4FE886B93CA62BD4A350D26BBA015287DF7E8DF2DD871AC23D`.
- Single mechanism: disable the reciprocal 2×2 ROAB route only for native `rows < channels`
  (`proj/down`) Linear matrices. The extra over-budget A3 residual pass is removed so the control
  has the same pre-A3 runtime envelope as its parent.
- Attention is unchanged and kept as the v86 implementation.

## Local targeted evidence

Protocol: `proxy-v2`, existing cache
`artifacts/official_eval/cache/qwen2.5-0.5b-proxy-v2.pt`, CPU/CUDA device as configured, 14 Linear
cases (two layers covering all seven roles) + 1 Attention case. This is a targeted smoke, not a
full-panel score.

| Candidate | Linear mean | Attention mean | Overall mean | API total | Wall |
|---|---:|---:|---:|---:|---:|
| pre-A3 parent | 0.582528216 | 0.942927486 | 0.606554834 | 201.258s | 209.078s |
| v151 | 0.582528216 | 0.942927486 | 0.606554834 | 193.213s | 199.430s |

Role means were identical for q/k/v/o/fc_gate/fc_up/proj. The lower local time is measurement noise
or an incidental branch difference, not an accepted runtime claim.

## External hif4 cross-check

Upstream `youxilee/hif4` `real_data_eval.py`, commit
`dd5ee6515323169dbd4133b3d4fd1ff1cb7be646`, GPT-2 small, `layers=4`, `seq=128`, `calib=2`,
`test=2`, `mode=amax6`, `config=current`, causal MHA:

| Candidate | q | k | v | o | fc | proj | Attention |
|---|---:|---:|---:|---:|---:|---:|---:|
| pre-A3 parent | 0.6802 | 0.7443 | 0.6505 | 0.7032 | 0.5658 | 0.5029 | 0.4540 |
| v151 | 0.6802 | 0.7443 | 0.6505 | 0.7032 | 0.5658 | 0.5658 | 0.4540 |

The external gain is isolated to `proj` (`+0.0629`), but it does not transfer to the canonical
Qwen panel. Full external protocol and limitations are in
`logs/execution/2026-09-02-v151-proj-roab-off.md`.

## Official result

- Score: `unregistered`.
- Time: `unregistered`.
- Status: `unregistered`.

## Decision

Do not change the root `solution.py` or use v151 as a parent. Keep the ROAB-off result as a
cross-model control; next implement an expansive `fc_gate/fc_up` encoder/scale mechanism while
retaining BOAT, then repeat the same role-differential smoke.

## Source

- SHA256: `65577F422E0DB1AAC9D3E27EC1DAB4EC5501FB8A6804C9349159268966CF8D25`
- Source: `solution.py`
