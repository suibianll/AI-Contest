# official-shape-v1 archive evaluation

- data source: `cache`
- capture seconds: `14.281`
- calibration lengths: `[10, 128, 512, 1024, 1024]`
- case counts: `250 Linear + 200 Attention`

| Candidate | Status | Linear mean | Attention mean | Equal-weight scale | API total(s) | API<300 | Wall(s) | Wall<300 | Official status |
|---|---|---:|---:|---:|---:|---|---:|---|---|
| v133-gain-adyn2-equivalent-idle-rerun-20260901 | ok | 0.483610 | 0.834256 | 28775.385 | 291.275 | True | 314.005 | False | unregistered |
