# proxy-v2 archive evaluation

- data source: `cache`
- capture seconds: `22.936`
- calibration lengths: `[10, 128, 512, 1024, 1024]`
- case counts: `25 Linear + 20 Attention` (5:4 trend ratio)

| Candidate | Status | Linear mean | Attention mean | Ratio mean | API total(s) | API calls | Wall(s) | Official status |
|---|---|---:|---:|---:|---:|---:|---:|---|
| v086-proxy-v2-dev | ok | 0.516784 | 0.727820 | 0.610578 | 77.215 | 130 | 79.666 | unregistered |
