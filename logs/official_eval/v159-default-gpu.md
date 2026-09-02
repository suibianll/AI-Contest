# proxy-v2 archive evaluation

- data source: `nvfp4_cache`
- evaluation scopes: `['unknown']`
- proxy ranking is valid only for identical `default-panel` cache/panel runs; effect-panel, full-stress, smoke-prefix and external probes are not ranking scores
- capture seconds: `1.718`
- calibration lengths: `[10, 128, 512, 1024, 1024]`
- case counts: `168 Linear + 120 Attention` (stratified real-W/A panel by default)
- official trend audit: `insufficient_anchors` (0 concordant / 0 inverted / 0 tied pairs)
- trend audit is a same-cohort diagnostic only; it never changes a proxy score
- error-source decomposition is stored per candidate in JSON `decomposition`/`case_scores`; archive table remains score-only

| Candidate | Status | Linear mean | Attention mean | Overall mean | API total(s) | API calls | Wall(s) | Official status |
|---|---|---:|---:|---:|---:|---:|---:|---|
| candidate | error | - | - | - | - | - | - | unregistered |
