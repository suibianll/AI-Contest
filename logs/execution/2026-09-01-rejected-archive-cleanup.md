# Rejected archive cleanup

To keep the version archive compact, the source directories for the rejected BDLR-JAQ family
v141–v145 were deleted after their experiments were closed:

- `solutions/20260901_v141_linear-bdlr-jaq-r4_scoreNA_timeNA`
- `solutions/20260901_v142_linear-bdlr-jaq-r4-anchorfreeze_scoreNA_timeNA`
- `solutions/20260901_v143_linear-bdlr-jaq-r4-dynamic-only_scoreNA_timeNA`
- `solutions/20260901_v144_linear-bdlr-jaq-r4-damped02_scoreNA_timeNA`
- `solutions/20260901_v145_linear-bdlr-jaq-r4-damped005_scoreNA_timeNA`

The per-run evaluator JSON files under `artifacts/official_eval/` and execution logs under
`logs/execution/` were retained. Their aggregate local results are Linear
`0.281760–0.506256`, Attention `0.715942`, API `204.681–211.460s`, and all are rejected because
they did not improve v140. No active root code was changed by this cleanup.
