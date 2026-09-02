# proxy-v2 archive evaluation

- data source: `cache`
- capture seconds: `51.198`
- calibration lengths: `[10, 128, 512, 1024, 1024]`
- case counts: `25 Linear + 20 Attention` (5:4 trend ratio)

| Candidate | Status | Linear mean | Attention mean | Ratio mean | API total(s) | API calls | Wall(s) | Official status |
|---|---|---:|---:|---:|---:|---:|---:|---|
| root | ok | 0.616425 | 0.725185 | 0.664763 | 87.811 | 130 | 92.941 | unregistered |
