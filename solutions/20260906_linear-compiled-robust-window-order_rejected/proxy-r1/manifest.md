# eval-v3 run

- candidate: `candidate`
- source SHA256: `9bbdaddbeb7fb1648758a127d5a14a742f37b9ff7b1ad50345d71e4139bd850c`
- scenario/OOD: `linear` / `False`
- shards: `[0, 1]`; stopped early: `False`
- overall local gain: `+0.634650`
- API total (diagnostic): `127.301s`; calibration cache hits: `0`
- official score/time equivalent: `false`

| side | cases | mean | median | min | max |
| --- | ---: | ---: | ---: | ---: | ---: |
| linear | 112 | +0.634650 | +0.597138 | +0.343435 | +0.931600 |
| attention | 0 | +0.000000 | +0.000000 | +0.000000 | +0.000000 |

## Checks

- source_exists: `True`
- source_sha256: `9bbdaddbeb7fb1648758a127d5a14a742f37b9ff7b1ad50345d71e4139bd850c`
- all_outputs_finite: `True`
- unique_case_identities: `True`
- expected_case_coverage: `True`
- official_score_equivalent: `False`

## Baseline

- source: `D:\工作内容\AI竞赛\solution.py`
- linear mean: `+0.634576` (112 cases)
- attention mean: `+0.000000` (0 cases)
