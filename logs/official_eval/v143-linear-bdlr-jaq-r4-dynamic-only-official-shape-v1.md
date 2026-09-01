# official-shape-v1 archive evaluation

- data source: `cache`
- capture seconds: `13.546`
- calibration lengths: `[10, 128, 512, 1024, 1024]`
- case counts: `250 Linear + 200 Attention`

| Candidate | Status | Linear mean | Attention mean | Equal-weight scale | API total(s) | API<300 | Wall(s) | Wall<300 | Official status |
|---|---|---:|---:|---:|---:|---|---:|---|---|
| v143-linear-bdlr-jaq-r4-dynamic-only-official-shape-v1 | ok | 0.361154 | 0.715942 | 23347.681 | 207.445 | True | 230.788 | True | unregistered |
