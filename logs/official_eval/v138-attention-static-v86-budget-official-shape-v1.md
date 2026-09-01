# official-shape-v1 archive evaluation

- data source: `cache`
- capture seconds: `14.632`
- calibration lengths: `[10, 128, 512, 1024, 1024]`
- case counts: `250 Linear + 200 Attention`

| Candidate | Status | Linear mean | Attention mean | Equal-weight scale | API total(s) | API<300 | Wall(s) | Wall<300 | Official status |
|---|---|---:|---:|---:|---:|---|---:|---|---|
| v138-attention-static-v86-budget-official-shape-v1 | ok | 0.507320 | 0.715942 | 27001.827 | 192.996 | True | 216.324 | True | unregistered |
