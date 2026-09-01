# official-shape-v1 archive evaluation

- data source: `cache`
- capture seconds: `16.990`
- calibration lengths: `[10, 128, 512, 1024, 1024]`
- case counts: `250 Linear + 200 Attention`

| Candidate | Status | Linear mean | Attention mean | Equal-weight scale | API total(s) | API<300 | Wall(s) | Wall<300 | Official status |
|---|---|---:|---:|---:|---:|---|---:|---|---|
| root-v127-output-weight-qwgram-gain-adyn2 | ok | 0.483610 | 0.834256 | 28775.385 | 365.818 | False | 397.341 | False | unregistered |
