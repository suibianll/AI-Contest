# 2026-09-08 current Linear + R3 Attention candidate

## Composition

- Linear: current root compiled sample-energy path, root source SHA256
  `D66128A62E7E068EDC50C91F4D8E212F586A6EDCAEE5BEA7D3564166E258B0F6`.
- Attention: R3 rotation-center/all-gates source SHA256
  `A5C679D7A2B349A879B2019B4A613244F5FEA7050E407B6475B9E29BD1C146DC`.
- Candidate source:
  `solutions/20260908_linear-current-r3-attention_candidate/solution.py`.
- Candidate source SHA256:
  `12352EFDD4E23CC5E1E17953008664FBAA5EA5D693373635FDAFC4D28CE4E24E`.

The candidate is an independent archive. Its final definitions keep R3's
Attention calibration and Q/K/V APIs while overriding the R3 side-isolation
Linear shadows with the current root Linear calibration and dynamic path. The
root `solution.py` remains unchanged.

## Verification

- Six public APIs, random-shape finite-output smoke, and reference state
  validation: passed.
- Proxy-v3 Qwen3.5-4B shard 0, scenario `both`: completed; records `1`;
  `reasonableness_issues=0`.
- Linear delta mean: `0.000000`; all 56 paired cases were zero, confirming
  the candidate carries the current root Linear behavior on this panel.
- Attention delta mean: `+0.0172405855`; L1 `0.0175740641`; 11 positive,
  1 negative, 0 zero. The one negative case was test length 10 at
  `-0.0020008718`.
- Overall delta mean: `+0.0030424563`.
- Local API total: `187.24s`; local timing is diagnostic and is not an
  official 300-second result.

The local analyzer result is `reject` for this screen because Linear has no
incremental effect (`delta_mean=0`); the Attention signal is positive but
mixed. No official score or time is registered. The additive estimate
`18032` and time-risk estimate `291s` remain prioritization diagnostics only.
