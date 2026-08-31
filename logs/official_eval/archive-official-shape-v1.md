# official-shape-v1 archive evaluation

- data source: `cache`
- capture seconds: `13.275`
- calibration lengths: `[10, 128, 512, 1024, 1024]`
- case counts: `250 Linear + 200 Attention`

| Candidate | Status | Linear mean | Attention mean | Equal-weight scale | API total(s) | API<300 | Wall(s) | Wall<300 | Official status |
|---|---|---:|---:|---:|---:|---|---:|---|---|
| v001 | ok | 0.315499 | 0.449961 | 16886.708 | 36.868 | True | 59.582 | True | pass |
| v002 | error | - | - | - | - | - | - | - | pass |
| v013 | ok | 0.437667 | 0.639206 | 23725.803 | 71.433 | True | 95.076 | True | pass |
| v024 | ok | 0.450075 | 0.639206 | 24035.985 | 76.148 | True | 99.192 | True | pass |
| v025 | ok | 0.387146 | 0.639206 | 22462.759 | 77.815 | True | 101.034 | True | pass |
| v030 | ok | 0.265321 | 0.639206 | 19417.153 | 93.005 | True | 116.100 | True | pass |
| v031 | ok | 0.374651 | 0.639206 | 22150.391 | 87.008 | True | 109.642 | True | pass |
| v032 | ok | 0.356338 | 0.639206 | 21692.574 | 143.127 | True | 165.348 | True | pass |
| v034 | ok | 0.374651 | 0.639206 | 22150.391 | 87.730 | True | 109.926 | True | pass |
| v051 | ok | 0.355097 | 0.639206 | 21661.535 | 163.484 | True | 185.502 | True | pass |
| v066 | ok | 0.357919 | 0.639242 | 21732.831 | 165.801 | True | 187.881 | True | pass |
| v072 | ok | 0.365587 | 0.639242 | 21924.524 | 165.616 | True | 187.545 | True | pass |
| v074 | ok | 0.372739 | 0.639242 | 22103.311 | 180.526 | True | 202.576 | True | pass |
| v084 | ok | 0.406668 | 0.718107 | 24528.845 | 279.191 | True | 300.848 | False | pass |
| v098 | ok | 0.465655 | 0.833573 | 28312.840 | 406.681 | False | 429.285 | False | timeout |
| v100 | ok | 0.465655 | 0.833617 | 28313.723 | 417.747 | False | 439.896 | False | wrong-answer/timeout |
| v107 | ok | 0.469211 | 0.833617 | 28402.621 | 436.719 | False | 459.727 | False | wrong-answer |
| v121 | ok | 0.472198 | 0.833617 | 28477.289 | 3404.369 | False | 3429.645 | False | timeout |
