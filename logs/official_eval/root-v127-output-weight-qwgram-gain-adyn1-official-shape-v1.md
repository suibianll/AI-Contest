# official-shape-v1 archive evaluation

- data source: `cache`
- capture seconds: `23.944`
- calibration lengths: `[10, 128, 512, 1024, 1024]`
- case counts: `250 Linear + 200 Attention`

| Candidate | Status | Linear mean | Attention mean | Equal-weight scale | API total(s) | API<300 | Wall(s) | Wall<300 | Official status |
|---|---|---:|---:|---:|---:|---|---:|---|---|
| root-v127-output-weight-qwgram-gain-adyn1 | ok | 0.476187 | 0.834256 | 28589.811 | 356.273 | False | 390.670 | False | unregistered |
