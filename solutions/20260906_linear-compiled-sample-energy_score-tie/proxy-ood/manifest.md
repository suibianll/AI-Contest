# eval-v3 run

- candidate: `compiled-sample-energy-ood`
- source SHA256: `f6541fadbac21e2876000729cf6a87a0761c7f796ff5fd65365c752207758626`
- scenario/OOD: `linear` / `True`
- shards: `[0, 1, 2, 3, 4, 5]`; stopped early: `False`
- overall local gain: `+0.652253`
- API total (diagnostic): `110.298s`; calibration cache hits: `6`
- official score/time equivalent: `false`

| side | cases | mean | median | min | max |
| --- | ---: | ---: | ---: | ---: | ---: |
| linear | 336 | +0.652253 | +0.652023 | +0.285644 | +0.988580 |
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
- linear mean: `+0.648735` (336 cases)
- attention mean: `+0.000000` (0 cases)
