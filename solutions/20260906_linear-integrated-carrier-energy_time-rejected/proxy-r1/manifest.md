# eval-v3 run

- candidate: `linear-integrated-carrier-energy`
- source SHA256: `a30e05b7fb71a64121908bc10d102dd50133a23e32f6568ed7044dbc45321730`
- scenario/OOD: `linear` / `False`
- shards: `[0, 1]`; stopped early: `False`
- overall local gain: `+0.634780`
- API total (diagnostic): `128.880s`; calibration cache hits: `0`
- official score/time equivalent: `false`

| side | cases | mean | median | min | max |
| --- | ---: | ---: | ---: | ---: | ---: |
| linear | 112 | +0.634780 | +0.598181 | +0.346453 | +0.931087 |
| attention | 0 | +0.000000 | +0.000000 | +0.000000 | +0.000000 |

## Checks

- source_exists: `True`
- source_sha256: `a30e05b7fb71a64121908bc10d102dd50133a23e32f6568ed7044dbc45321730`
- all_outputs_finite: `True`
- unique_case_identities: `True`
- expected_case_coverage: `True`
- official_score_equivalent: `False`

## Baseline

- source: `D:\工作内容\AI竞赛\solution.py`
- linear mean: `+0.634576` (112 cases)
- attention mean: `+0.000000` (0 cases)
