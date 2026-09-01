# official-shape-v1 archive evaluation

- data source: `cache`
- capture seconds: `14.338`
- calibration lengths: `[10, 128, 512, 1024, 1024]`
- case counts: `250 Linear + 200 Attention`

| Candidate | Status | Linear mean | Attention mean | Equal-weight scale | API total(s) | API<300 | Wall(s) | Wall<300 | Official status |
|---|---|---:|---:|---:|---:|---|---:|---|---|
| v136-linear-output-jacobi-dynamic-only-official-shape-v1 | ok | 0.500132 | 0.834256 | 29188.439 | 287.816 | True | 310.472 | False | unregistered |
