# official-shape-v1 archive evaluation

- data source: `cache`
- capture seconds: `16.058`
- calibration lengths: `[10, 128, 512, 1024, 1024]`
- case counts: `250 Linear + 200 Attention`

| Candidate | Status | Linear mean | Attention mean | Equal-weight scale | API total(s) | API<300 | Wall(s) | Wall<300 | Official status |
|---|---|---:|---:|---:|---:|---|---:|---|---|
| v148-joint-wa-v86-attention-v140-linear | ok | 0.509729 | 0.719696 | 27137.139 | 369.038 | False | 391.615 | False | unregistered |
