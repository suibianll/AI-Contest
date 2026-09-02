# proxy-v2 archive evaluation

- data source: `cache`
- capture seconds: `23.060`
- calibration lengths: `[10, 128, 512, 1024, 1024]`
- case counts: `25 Linear + 20 Attention` (5:4 trend ratio)

| Candidate | Status | Linear mean | Attention mean | Ratio mean | API total(s) | API calls | Wall(s) | Official status |
|---|---|---:|---:|---:|---:|---:|---:|---|
| v138-proxy-v2-dev | ok | 0.609434 | 0.737584 | 0.666389 | 52.271 | 130 | 54.861 | unregistered |
