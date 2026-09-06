# eval-v3 run

- candidate: `candidate`
- source SHA256: `3ca71a9d5edc1833a5b62f0ffaf0afa6156197f802585c58ceeb160f078b1189`
- scenario/OOD: `linear` / `False`
- shards: `[0, 1, 2, 3, 4, 5]`; stopped early: `False`
- overall local gain: `+0.639781`
- API total (diagnostic): `286.630s`; calibration cache hits: `2`
- official score/time equivalent: `false`

| side | cases | mean | median | min | max |
| --- | ---: | ---: | ---: | ---: | ---: |
| linear | 336 | +0.639781 | +0.637249 | +0.172379 | +0.930954 |
| attention | 0 | +0.000000 | +0.000000 | +0.000000 | +0.000000 |

## Checks

- source_exists: `True`
- source_sha256: `3ca71a9d5edc1833a5b62f0ffaf0afa6156197f802585c58ceeb160f078b1189`
- all_outputs_finite: `True`
- unique_case_identities: `True`
- expected_case_coverage: `True`
- official_score_equivalent: `False`

## Baseline

- source: `D:\工作内容\AI竞赛\solution.py`
- linear mean: `+0.636799` (336 cases)
- attention mean: `+0.000000` (0 cases)
