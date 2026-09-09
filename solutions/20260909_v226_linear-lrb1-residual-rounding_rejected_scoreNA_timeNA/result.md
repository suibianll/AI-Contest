# v226 — L-RB1 静态权重输出残差共享舍入边界

Status: `REJECTED` locally; official status `unregistered/NA`. No official submission was made.

- Parent: retained v202 complete root, SHA256 `56dc805d6e5a3aef896db8021045740292735725d688b48e3d4393e55efcb2bd`.
- Candidate SHA256: `31d08ce90140e1527b5ffbff9e92e27d87b9d25d6239b0560daf65f6ed468e3d`.
- Mechanism: replace the fixed nearest-rounding threshold `0.5` of the static weight quantizer by one learned
  boundary per `(sign, lower-code)` class — 12 shared scalars per calibration call — with
  `c = m if frac(u) < tau[s,m] else m+1`, `m = floor(u)`. Only elements whose stored parent code equals the
  plain nearest rounding of the deployed-coordinate magnitude enter the fit, so `tau = 0.5` restores the
  parent five fields bit for bit. Boundaries are solved once from the frozen parent A@W output residual in
  64 fixed fractional buckets, merged into one table, and accepted only when the exact parent-coordinate
  quadratic output loss strictly decreases.
- Fit: all calibration pairs and rows supplied by eval-v3, case-equal MSE; no parameter sweep, no official result awaited.

## Local result

`eval-v3`, Qwen3.5-4B proxy-v2, Linear-only target side, all six shards (336 cases):

| shard | delta gain mean | median | positive / negative / zero |
|---:|---:|---:|---|
| 0 | -0.000143631 | 0.0 | 7 / 12 / 37 |
| 1 | -0.000593319 | 0.0 | 5 / 19 / 32 |
| 2 | -0.000339417 | 0.0 | 7 / 17 / 32 |
| 3 | -0.000439196 | 0.0 | 6 / 13 / 37 |
| 4 | -0.000599656 | 0.0 | 4 / 16 / 36 |
| 5 | -0.000346331 | 0.0 | 4 / 18 / 34 |

Equal-shard mean delta gain `-0.000410258`; pooled `33 / 95 / 208` of 336 cases; every shard decided `reject`.

Per-layer diagnostics (168 calibration calls across the six shards):

| metric | value |
|---|---:|
| accepted layers / reverted layers | 65 / 103 |
| boundary proposals | 334 |
| changed mantissa codes | 10,381,928 |
| accepted calibration `ΔL` sum | -2.519897e-05 |
| reverted calibration `ΔL` sum | +8.270461e-04 |

All 103 reverts were caused by `ΔL >= 0`, so the solver and the acceptance gate behaved as designed; the
mechanism was reachable and non-equivalent. Accepted layers concentrate in the wide shapes
(`2560x9216`, `2560x4096`, `9216x2560`, `4096x2560`), which are also the worst-case shapes of the paired run.

Timing (diagnostic only, not a gate): candidate six-shard fresh `api_total` 1289.2s and calibration 1018.0s;
the parent shard0 fresh control is 179.3s / 134.0s against the candidate's 204.6s / 164.2s
(≈1.14x `api_total`, ≈1.23x calibration).

## Interpretation

The shared boundary table can buy at most ~5e-7 of exact calibration A@W loss per accepted layer while
flipping 10^5-10^6 hard codes. That is the "high-degree-of-freedom per-code edit overfits the calibration
window" shape the active plan identified in section 1, not the "few shared boundaries decide many real hard
codes" shape the card was meant to test: the loss is at the numerical noise floor and the perturbation is not.
The mechanism is closed as `REJECTED`; per the plan it is not retried with per-row/per-block tables, more
buckets, sign splits or other threshold granularity. The root remains v202 Linear + v195 Attention at `18053/281s`.
