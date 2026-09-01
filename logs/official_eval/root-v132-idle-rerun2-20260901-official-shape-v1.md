# official-shape-v1 archive evaluation

- data source: `cache`
- capture seconds: `15.440`
- calibration lengths: `[10, 128, 512, 1024, 1024]`
- case counts: `250 Linear + 200 Attention`

| Candidate | Status | Linear mean | Attention mean | Equal-weight scale | API total(s) | API<300 | Wall(s) | Wall<300 | Official status |
|---|---|---:|---:|---:|---:|---|---:|---|---|
| v132-root-idle-rerun2-20260901 | ok | 0.473131 | 0.834256 | 28513.390 | 289.318 | True | 311.974 | False | unregistered |
