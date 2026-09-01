# official-shape-v1 archive evaluation

- data source: `cache`
- capture seconds: `15.834`
- calibration lengths: `[10, 128, 512, 1024, 1024]`
- case counts: `250 Linear + 200 Attention`

| Candidate | Status | Linear mean | Attention mean | Equal-weight scale | API total(s) | API<300 | Wall(s) | Wall<300 | Official status |
|---|---|---:|---:|---:|---:|---|---:|---|---|
| v140-linear-roab-pair-official-shape-v1 | ok | 0.507355 | 0.715942 | 27002.705 | 205.365 | True | 229.337 | True | unregistered |
