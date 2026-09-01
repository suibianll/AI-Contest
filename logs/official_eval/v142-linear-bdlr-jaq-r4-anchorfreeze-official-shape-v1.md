# official-shape-v1 archive evaluation

- data source: `cache`
- capture seconds: `15.153`
- calibration lengths: `[10, 128, 512, 1024, 1024]`
- case counts: `250 Linear + 200 Attention`

| Candidate | Status | Linear mean | Attention mean | Equal-weight scale | API total(s) | API<300 | Wall(s) | Wall<300 | Official status |
|---|---|---:|---:|---:|---:|---|---:|---|---|
| v142-linear-bdlr-jaq-r4-anchorfreeze-official-shape-v1 | ok | 0.282559 | 0.715942 | 21382.822 | 211.460 | True | 234.842 | True | unregistered |
