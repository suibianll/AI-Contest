# official-shape-v1 archive evaluation

- data source: `cache`
- capture seconds: `14.004`
- calibration lengths: `[10, 128, 512, 1024, 1024]`
- case counts: `250 Linear + 200 Attention`

| Candidate | Status | Linear mean | Attention mean | Equal-weight scale | API total(s) | API<300 | Wall(s) | Wall<300 | Official status |
|---|---|---:|---:|---:|---:|---|---:|---|---|
| root-v127-fixed-attn-budget | ok | 0.465655 | 0.837789 | 28397.163 | 310.732 | False | 332.557 | False | unregistered |
