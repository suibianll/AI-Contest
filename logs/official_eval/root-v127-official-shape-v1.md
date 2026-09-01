# official-shape-v1 archive evaluation

- data source: `cache`
- capture seconds: `15.182`
- calibration lengths: `[10, 128, 512, 1024, 1024]`
- case counts: `250 Linear + 200 Attention`

| Candidate | Status | Linear mean | Attention mean | Equal-weight scale | API total(s) | API<300 | Wall(s) | Wall<300 | Official status |
|---|---|---:|---:|---:|---:|---|---:|---|---|
| root-v127 | ok | 0.465655 | 0.833617 | 28313.723 | 416.465 | False | 439.617 | False | unregistered |
