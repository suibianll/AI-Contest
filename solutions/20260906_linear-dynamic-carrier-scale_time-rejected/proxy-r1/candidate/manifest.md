# eval-v3 run

- candidate: `candidate`
- source SHA256: `a9c8ce0eb2649c0b968b46a0f8fa41f88c13b214779d936f67da7a4f9db3368d`
- scenario/OOD: `linear` / `False`
- shards: `[0, 1]`; stopped early: `False`
- overall local gain: `+0.635186`
- API total (diagnostic): `127.156s`; calibration cache hits: `0`
- official score/time equivalent: `false`

| side | cases | mean | median | min | max |
| --- | ---: | ---: | ---: | ---: | ---: |
| linear | 112 | +0.635186 | +0.602215 | +0.348548 | +0.930811 |
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
- linear mean: `+0.634576` (112 cases)
- attention mean: `+0.000000` (0 cases)
