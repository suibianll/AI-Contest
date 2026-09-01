# official-shape-v1 archive evaluation

- data source: `cache`
- capture seconds: `14.458`
- calibration lengths: `[10, 128, 512, 1024, 1024]`
- case counts: `250 Linear + 200 Attention`

| Candidate | Status | Linear mean | Attention mean | Equal-weight scale | API total(s) | API<300 | Wall(s) | Wall<300 | Official status |
|---|---|---:|---:|---:|---:|---|---:|---|---|
| v147-v86-attention-v140-linear | ok | 0.507355 | 0.719696 | 27077.787 | 222.227 | True | 245.038 | True | unregistered |
