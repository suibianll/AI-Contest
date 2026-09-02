# proxy-v2 archive evaluation

- data source: `cache`
- capture seconds: `56.063`
- calibration lengths: `[10, 128, 512, 1024, 1024]`
- case counts: `25 Linear + 20 Attention` (5:4 trend ratio)

| Candidate | Status | Linear mean | Attention mean | Ratio mean | API total(s) | API calls | Wall(s) | Official status |
|---|---|---:|---:|---:|---:|---:|---:|---|
| v086-proxy-v2-variable-dev | ok | 0.519719 | 0.725185 | 0.611037 | 84.449 | 130 | 89.694 | unregistered |
