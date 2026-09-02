# proxy-v2 archive evaluation

- data source: `cache`
- capture seconds: `48.357`
- calibration lengths: `[10, 128, 512, 1024, 1024]`
- case counts: `25 Linear + 20 Attention` (5:4 trend ratio)

| Candidate | Status | Linear mean | Attention mean | Ratio mean | API total(s) | API calls | Wall(s) | Official status |
|---|---|---:|---:|---:|---:|---:|---:|---|
| v138-proxy-v2-variable-dev | ok | 0.616425 | 0.742170 | 0.672312 | 58.049 | 130 | 63.600 | unregistered |
