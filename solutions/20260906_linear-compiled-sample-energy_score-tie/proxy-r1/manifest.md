# eval-v3 run

- candidate: `compiled-sample-energy-r1`
- source SHA256: `f6541fadbac21e2876000729cf6a87a0761c7f796ff5fd65365c752207758626`
- scenario/OOD: `linear` / `False`
- shards: `[0, 1]`; stopped early: `False`
- overall local gain: `+0.636504`
- API total (diagnostic): `123.373s`; calibration cache hits: `0`
- official score/time equivalent: `false`

| side | cases | mean | median | min | max |
| --- | ---: | ---: | ---: | ---: | ---: |
| linear | 112 | +0.636504 | +0.601644 | +0.351000 | +0.930954 |
| attention | 0 | +0.000000 | +0.000000 | +0.000000 | +0.000000 |

## Checks

- source_exists: `True`
- source_sha256: `f6541fadbac21e2876000729cf6a87a0761c7f796ff5fd65365c752207758626`
- all_outputs_finite: `True`
- unique_case_identities: `True`
- expected_case_coverage: `True`
- official_score_equivalent: `False`

## Baseline

- source: `D:\工作内容\AI竞赛\solution.py`
- linear mean: `+0.634576` (112 cases)
- attention mean: `+0.000000` (0 cases)
