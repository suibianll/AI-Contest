# official-shape-v1 archive evaluation

- data source: `cache`
- capture seconds: `14.508`
- calibration lengths: `[10, 128, 512, 1024, 1024]`
- case counts: `250 Linear + 200 Attention`

| Candidate | Status | Linear mean | Attention mean | Equal-weight scale | API total(s) | API<300 | Wall(s) | Wall<300 | Official status |
|---|---|---:|---:|---:|---:|---|---:|---|---|
| active-v140-linear-v86-attention-a3 | ok | 0.510050 | 0.719696 | 27145.179 | 300.351 | False | 325.313 | False | unregistered |
