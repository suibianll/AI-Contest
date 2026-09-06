# eval-v3 run

- candidate: `compiled-output-covariance-r1`
- source SHA256: `a6008b91ef2be558d2a53bf4cb4ce31912344243a38f71aeb8cbeb7d3af439ea`
- scenario/OOD: `linear` / `False`
- shards: `[0, 1]`; stopped early: `True`
- overall local gain: `+0.634332`
- API total (diagnostic): `124.573s`; calibration cache hits: `0`
- official score/time equivalent: `false`

| side | cases | mean | median | min | max |
| --- | ---: | ---: | ---: | ---: | ---: |
| linear | 112 | +0.634332 | +0.596826 | +0.346979 | +0.930927 |
| attention | 0 | +0.000000 | +0.000000 | +0.000000 | +0.000000 |

## Checks

- source_exists: `True`
- source_sha256: `a6008b91ef2be558d2a53bf4cb4ce31912344243a38f71aeb8cbeb7d3af439ea`
- all_outputs_finite: `True`
- unique_case_identities: `True`
- expected_case_coverage: `None`
- official_score_equivalent: `False`

## Baseline

- source: `D:\工作内容\AI竞赛\solution.py`
- linear mean: `+0.634576` (112 cases)
- attention mean: `+0.000000` (0 cases)
