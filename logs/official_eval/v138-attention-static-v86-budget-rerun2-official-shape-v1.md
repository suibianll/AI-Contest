# official-shape-v1 archive evaluation

- data source: `cache`
- capture seconds: `14.871`
- calibration lengths: `[10, 128, 512, 1024, 1024]`
- case counts: `250 Linear + 200 Attention`

| Candidate | Status | Linear mean | Attention mean | Equal-weight scale | API total(s) | API<300 | Wall(s) | Wall<300 | Official status |
|---|---|---:|---:|---:|---:|---|---:|---|---|
| v138-attention-static-v86-budget-rerun2 | ok | 0.507320 | 0.715942 | 27001.827 | 187.935 | True | 210.855 | True | unregistered |
