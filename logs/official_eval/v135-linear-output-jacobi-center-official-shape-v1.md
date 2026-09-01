# official-shape-v1 archive evaluation

- data source: `cache`
- capture seconds: `14.148`
- calibration lengths: `[10, 128, 512, 1024, 1024]`
- case counts: `250 Linear + 200 Attention`

| Candidate | Status | Linear mean | Attention mean | Equal-weight scale | API total(s) | API<300 | Wall(s) | Wall<300 | Official status |
|---|---|---:|---:|---:|---:|---|---:|---|---|
| v135-linear-output-jacobi-center-official-shape-v1 | ok | 0.500812 | 0.834256 | 29205.431 | 290.823 | True | 313.365 | False | unregistered |
