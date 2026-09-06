# eval-v3 run

- candidate: `linear-integrated-carrier-energy`
- source SHA256: `a30e05b7fb71a64121908bc10d102dd50133a23e32f6568ed7044dbc45321730`
- scenario/OOD: `linear` / `False`
- shards: `[0, 1, 2, 3, 4, 5]`; stopped early: `False`
- overall local gain: `+0.637750`
- API total (diagnostic): `286.530s`; calibration cache hits: `2`
- official score/time equivalent: `false`

| side | cases | mean | median | min | max |
| --- | ---: | ---: | ---: | ---: | ---: |
| linear | 336 | +0.637750 | +0.633152 | +0.159834 | +0.931087 |
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
- linear mean: `+0.636799` (336 cases)
- attention mean: `+0.000000` (0 cases)
