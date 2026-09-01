# official-shape-v1 archive evaluation

- data source: `cache`
- capture seconds: `14.105`
- calibration lengths: `[10, 128, 512, 1024, 1024]`
- case counts: `250 Linear + 200 Attention`

| Candidate | Status | Linear mean | Attention mean | Equal-weight scale | API total(s) | API<300 | Wall(s) | Wall<300 | Official status |
|---|---|---:|---:|---:|---:|---|---:|---|---|
| v086-idle-rerun-20260901 | ok | 0.406668 | 0.719696 | 24560.627 | 299.302 | True | 321.996 | False | unregistered |
