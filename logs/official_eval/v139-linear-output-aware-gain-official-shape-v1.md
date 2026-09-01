# official-shape-v1 archive evaluation

- data source: `cache`
- capture seconds: `14.291`
- calibration lengths: `[10, 128, 512, 1024, 1024]`
- case counts: `250 Linear + 200 Attention`

| Candidate | Status | Linear mean | Attention mean | Equal-weight scale | API total(s) | API<300 | Wall(s) | Wall<300 | Official status |
|---|---|---:|---:|---:|---:|---|---:|---|---|
| v139-linear-output-aware-gain-official-shape-v1 | ok | 0.507278 | 0.715942 | 27000.796 | 193.389 | True | 217.196 | True | unregistered |
