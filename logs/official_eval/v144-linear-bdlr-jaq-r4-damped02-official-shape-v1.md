# official-shape-v1 archive evaluation

- data source: `cache`
- capture seconds: `13.747`
- calibration lengths: `[10, 128, 512, 1024, 1024]`
- case counts: `250 Linear + 200 Attention`

| Candidate | Status | Linear mean | Attention mean | Equal-weight scale | API total(s) | API<300 | Wall(s) | Wall<300 | Official status |
|---|---|---:|---:|---:|---:|---|---:|---|---|
| v144-linear-bdlr-jaq-r4-damped02-official-shape-v1 | ok | 0.506418 | 0.715942 | 26979.286 | 208.414 | True | 232.178 | True | unregistered |
