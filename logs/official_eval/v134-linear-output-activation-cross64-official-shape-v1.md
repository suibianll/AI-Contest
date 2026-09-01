# official-shape-v1 archive evaluation

- data source: `cache`
- capture seconds: `14.740`
- calibration lengths: `[10, 128, 512, 1024, 1024]`
- case counts: `250 Linear + 200 Attention`

| Candidate | Status | Linear mean | Attention mean | Equal-weight scale | API total(s) | API<300 | Wall(s) | Wall<300 | Official status |
|---|---|---:|---:|---:|---:|---|---:|---|---|
| v134-linear-output-activation-cross64 | ok | 0.507320 | 0.834256 | 29368.117 | 289.042 | True | 312.315 | False | unregistered |
