# official-shape-v1 archive evaluation

- data source: `cache`
- capture seconds: `15.825`
- calibration lengths: `[10, 128, 512, 1024, 1024]`
- case counts: `250 Linear + 200 Attention`

| Candidate | Status | Linear mean | Attention mean | Equal-weight scale | API total(s) | API<300 | Wall(s) | Wall<300 | Official status |
|---|---|---:|---:|---:|---:|---|---:|---|---|
| v141-linear-bdlr-jaq-r4-official-shape-v1 | ok | 0.281760 | 0.715942 | 21362.836 | 204.681 | True | 228.127 | True | unregistered |
