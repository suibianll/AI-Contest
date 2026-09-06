# eval-v3 run

- candidate: `linear-dynamic-block-energy`
- source SHA256: `fba242715fc731f658a2dae1bb8375a81e3862bfc538fb8f711162a6c03e7256`
- scenario/OOD: `linear` / `False`
- shards: `[0, 1, 2, 3, 4, 5]`; stopped early: `False`
- overall local gain: `+0.639811`
- API total (diagnostic): `286.276s`; calibration cache hits: `2`
- official score/time equivalent: `false`

| side | cases | mean | median | min | max |
| --- | ---: | ---: | ---: | ---: | ---: |
| linear | 336 | +0.639811 | +0.637259 | +0.174019 | +0.930571 |
| attention | 0 | +0.000000 | +0.000000 | +0.000000 | +0.000000 |

## Checks

- source_exists: `True`
- source_sha256: `fba242715fc731f658a2dae1bb8375a81e3862bfc538fb8f711162a6c03e7256`
- all_outputs_finite: `True`
- unique_case_identities: `True`
- expected_case_coverage: `True`
- official_score_equivalent: `False`

## Baseline

- source: `D:\工作内容\AI竞赛\solution.py`
- linear mean: `+0.636799` (336 cases)
- attention mean: `+0.000000` (0 cases)
