# official-shape-v1 archive evaluation

- data source: `cache`
- capture seconds: `14.296`
- calibration lengths: `[10, 128, 512, 1024, 1024]`
- case counts: `250 Linear + 200 Attention`

| Candidate | Status | Linear mean | Attention mean | Equal-weight scale | API total(s) | API<300 | Wall(s) | Wall<300 | Official status |
|---|---|---:|---:|---:|---:|---|---:|---|---|
| root-v127-output-weight-qwgram | ok | 0.473131 | 0.836579 | 28559.836 | 294.835 | True | 317.708 | False | unregistered |
