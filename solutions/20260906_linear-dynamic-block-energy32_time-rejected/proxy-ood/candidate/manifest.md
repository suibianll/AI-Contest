# eval-v3 run

- candidate: `candidate`
- source SHA256: `325f9bbe616ab6f308f190c9b4b9cce4f767f9ea5e9c94895177700f51a02e6e`
- scenario/OOD: `linear` / `True`
- shards: `[0, 1, 2, 3, 4, 5]`; stopped early: `False`
- overall local gain: `+0.651962`
- API total (diagnostic): `113.115s`; calibration cache hits: `6`
- official score/time equivalent: `false`

| side | cases | mean | median | min | max |
| --- | ---: | ---: | ---: | ---: | ---: |
| linear | 336 | +0.651962 | +0.651707 | +0.286035 | +0.988673 |
| attention | 0 | +0.000000 | +0.000000 | +0.000000 | +0.000000 |

## Checks

- source_exists: `True`
- source_sha256: `325f9bbe616ab6f308f190c9b4b9cce4f767f9ea5e9c94895177700f51a02e6e`
- all_outputs_finite: `True`
- unique_case_identities: `True`
- expected_case_coverage: `True`
- official_score_equivalent: `False`

## Baseline

- source: `D:\工作内容\AI竞赛\solution.py`
- linear mean: `+0.648735` (336 cases)
- attention mean: `+0.000000` (0 cases)
