# eval-v3 run

- candidate: `candidate`
- source SHA256: `c382bd4fe4acdbb21b4aacd6b009051093d3fd92a264ae645956f96def26958c`
- scenario/OOD: `linear` / `True`
- shards: `[0, 1, 2, 3, 4, 5]`; stopped early: `False`
- overall local gain: `+0.649639`
- API total (diagnostic): `109.466s`; calibration cache hits: `6`
- official score/time equivalent: `false`

| side | cases | mean | median | min | max |
| --- | ---: | ---: | ---: | ---: | ---: |
| linear | 336 | +0.649639 | +0.648025 | +0.267676 | +0.988585 |
| attention | 0 | +0.000000 | +0.000000 | +0.000000 | +0.000000 |

## Checks

- source_exists: `True`
- source_sha256: `c382bd4fe4acdbb21b4aacd6b009051093d3fd92a264ae645956f96def26958c`
- all_outputs_finite: `True`
- unique_case_identities: `True`
- expected_case_coverage: `True`
- official_score_equivalent: `False`

## Baseline

- source: `D:\工作内容\AI竞赛\workbench\linear_static_actorder_hdiag_recovered_solution.py`
- linear mean: `+0.648735` (336 cases)
- attention mean: `+0.000000` (0 cases)
