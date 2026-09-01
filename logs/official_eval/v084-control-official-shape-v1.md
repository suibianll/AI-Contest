# official-shape-v1 archive evaluation

- data source: `cache`
- capture seconds: `21.119`
- calibration lengths: `[10, 128, 512, 1024, 1024]`
- case counts: `250 Linear + 200 Attention`

| Candidate | Status | Linear mean | Attention mean | Equal-weight scale | API total(s) | API<300 | Wall(s) | Wall<300 | Official status |
|---|---|---:|---:|---:|---:|---|---:|---|---|
| v084-control | ok | 0.406668 | 0.718107 | 24528.845 | 458.740 | False | 484.842 | False | unregistered |
