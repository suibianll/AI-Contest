# eval-v3 run

- candidate: `candidate`
- source SHA256: `a9c8ce0eb2649c0b968b46a0f8fa41f88c13b214779d936f67da7a4f9db3368d`
- scenario/OOD: `linear` / `True`
- shards: `[0, 1, 2, 3, 4, 5]`; stopped early: `False`
- overall local gain: `+0.650456`
- API total (diagnostic): `113.879s`; calibration cache hits: `6`
- official score/time equivalent: `false`

| side | cases | mean | median | min | max |
| --- | ---: | ---: | ---: | ---: | ---: |
| linear | 336 | +0.650456 | +0.650046 | +0.276780 | +0.988655 |
| attention | 0 | +0.000000 | +0.000000 | +0.000000 | +0.000000 |

## Checks

- source_exists: `True`
- source_sha256: `a9c8ce0eb2649c0b968b46a0f8fa41f88c13b214779d936f67da7a4f9db3368d`
- all_outputs_finite: `True`
- unique_case_identities: `True`
- expected_case_coverage: `True`
- official_score_equivalent: `False`

## Baseline

- source: `D:\工作内容\AI竞赛\solution.py`
- linear mean: `+0.648735` (336 cases)
- attention mean: `+0.000000` (0 cases)
