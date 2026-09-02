# proxy-v2 archive evaluation

- data source: `cache`
- capture seconds: `23.673`
- calibration lengths: `[10, 128, 512, 1024, 1024]`
- case counts: `5 Linear + 4 Attention` (5:4 trend ratio)

| Candidate | Status | Linear mean | Attention mean | Ratio mean | API total(s) | API calls | Wall(s) | Official status |
|---|---|---:|---:|---:|---:|---:|---:|---|
| proxy-v2-smoke | ok | 0.706495 | 0.721345 | 0.713095 | 17.730 | 26 | 18.126 | unregistered |
