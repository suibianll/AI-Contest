# eval-v3 run

- candidate: `compiled-output-covariance-r1-repair`
- source SHA256: `52de271400805249ea0566f963eae96df61ce65af6fea5db4d76a83dd0782f02`
- scenario/OOD: `linear` / `False`
- shards: `[0, 1]`; stopped early: `False`
- overall local gain: `+0.634804`
- API total (diagnostic): `124.362s`; calibration cache hits: `0`
- official score/time equivalent: `false`

| side | cases | mean | median | min | max |
| --- | ---: | ---: | ---: | ---: | ---: |
| linear | 112 | +0.634804 | +0.597700 | +0.346979 | +0.930927 |
| attention | 0 | +0.000000 | +0.000000 | +0.000000 | +0.000000 |

## Checks

- source_exists: `True`
- source_sha256: `52de271400805249ea0566f963eae96df61ce65af6fea5db4d76a83dd0782f02`
- all_outputs_finite: `True`
- unique_case_identities: `True`
- expected_case_coverage: `True`
- official_score_equivalent: `False`

## Baseline

- source: `D:\工作内容\AI竞赛\solution.py`
- linear mean: `+0.634576` (112 cases)
- attention mean: `+0.000000` (0 cases)
