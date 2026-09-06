# eval-v3 run

- candidate: `compiled-output-covariance-ood`
- source SHA256: `52de271400805249ea0566f963eae96df61ce65af6fea5db4d76a83dd0782f02`
- scenario/OOD: `linear` / `True`
- shards: `[0, 1, 2, 3, 4, 5]`; stopped early: `False`
- overall local gain: `+0.649420`
- API total (diagnostic): `110.223s`; calibration cache hits: `6`
- official score/time equivalent: `false`

| side | cases | mean | median | min | max |
| --- | ---: | ---: | ---: | ---: | ---: |
| linear | 336 | +0.649420 | +0.649564 | +0.252687 | +0.987316 |
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
- linear mean: `+0.648735` (336 cases)
- attention mean: `+0.000000` (0 cases)
