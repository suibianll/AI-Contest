# official-shape-v1 archive evaluation

- data source: `cache`
- capture seconds: `14.941`
- calibration lengths: `[10, 128, 512, 1024, 1024]`
- case counts: `250 Linear + 200 Attention`

| Candidate | Status | Linear mean | Attention mean | Equal-weight scale | API total(s) | API<300 | Wall(s) | Wall<300 | Official status |
|---|---|---:|---:|---:|---:|---|---:|---|---|
| root-v127-no-pawv | ok | 0.465655 | 0.833573 | 28312.840 | 405.851 | False | 428.122 | False | unregistered |
