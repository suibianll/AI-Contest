# proxy-v2 archive evaluation

- data source: `cache`
- capture seconds: `53.132`
- calibration lengths: `[10, 128, 512, 1024, 1024]`
- case counts: `2016 Linear + 288 Attention` (all captured W/A by default)
- official trend audit: `insufficient_anchors` (0 concordant / 0 inverted / 0 tied pairs)
- trend audit is a same-cohort diagnostic only; it never changes a proxy score

| Candidate | Status | Linear mean | Attention mean | Overall mean | API total(s) | API calls | Wall(s) | Official status |
|---|---|---:|---:|---:|---:|---:|---:|---|
