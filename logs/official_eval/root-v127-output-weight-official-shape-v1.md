# official-shape-v1 archive evaluation

- data source: `cache`
- capture seconds: `14.456`
- calibration lengths: `[10, 128, 512, 1024, 1024]`
- case counts: `250 Linear + 200 Attention`

| Candidate | Status | Linear mean | Attention mean | Equal-weight scale | API total(s) | API<300 | Wall(s) | Wall<300 | Official status |
|---|---|---:|---:|---:|---:|---|---:|---|---|
| root-v127-output-weight | ok | 0.471837 | 0.836579 | 28527.508 | 295.437 | True | 317.607 | False | unregistered |
