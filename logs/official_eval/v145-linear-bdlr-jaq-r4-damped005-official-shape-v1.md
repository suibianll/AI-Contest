# official-shape-v1 archive evaluation

- data source: `cache`
- capture seconds: `14.073`
- calibration lengths: `[10, 128, 512, 1024, 1024]`
- case counts: `250 Linear + 200 Attention`

| Candidate | Status | Linear mean | Attention mean | Equal-weight scale | API total(s) | API<300 | Wall(s) | Wall<300 | Official status |
|---|---|---:|---:|---:|---:|---|---:|---|---|
| v145-linear-bdlr-jaq-r4-damped005-official-shape-v1 | ok | 0.506256 | 0.715942 | 26975.229 | 208.513 | True | 232.206 | True | unregistered |
