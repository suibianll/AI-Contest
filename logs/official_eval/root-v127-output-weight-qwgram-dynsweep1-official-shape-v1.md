# official-shape-v1 archive evaluation

- data source: `cache`
- capture seconds: `14.449`
- calibration lengths: `[10, 128, 512, 1024, 1024]`
- case counts: `250 Linear + 200 Attention`

| Candidate | Status | Linear mean | Attention mean | Equal-weight scale | API total(s) | API<300 | Wall(s) | Wall<300 | Official status |
|---|---|---:|---:|---:|---:|---|---:|---|---|
| root-v127-output-weight-qwgram-dynsweep1 | ok | 0.473131 | 0.828014 | 28388.545 | 273.792 | True | 295.958 | True | unregistered |
