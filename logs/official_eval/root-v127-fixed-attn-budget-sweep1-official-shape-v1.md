# official-shape-v1 archive evaluation

- data source: `cache`
- capture seconds: `13.472`
- calibration lengths: `[10, 128, 512, 1024, 1024]`
- case counts: `250 Linear + 200 Attention`

| Candidate | Status | Linear mean | Attention mean | Equal-weight scale | API total(s) | API<300 | Wall(s) | Wall<300 | Official status |
|---|---|---:|---:|---:|---:|---|---:|---|---|
| root-v127-fixed-attn-budget-sweep1 | ok | 0.465655 | 0.836579 | 28372.949 | 248.363 | True | 270.606 | True | unregistered |
