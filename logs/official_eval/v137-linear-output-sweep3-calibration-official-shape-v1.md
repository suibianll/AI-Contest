# official-shape-v1 archive evaluation

- data source: `cache`
- capture seconds: `14.295`
- calibration lengths: `[10, 128, 512, 1024, 1024]`
- case counts: `250 Linear + 200 Attention`

| Candidate | Status | Linear mean | Attention mean | Equal-weight scale | API total(s) | API<300 | Wall(s) | Wall<300 | Official status |
|---|---|---:|---:|---:|---:|---|---:|---|---|
| v137-linear-output-sweep3-calibration-official-shape-v1 | ok | 0.507163 | 0.834256 | 29364.202 | 296.755 | True | 319.306 | False | unregistered |
