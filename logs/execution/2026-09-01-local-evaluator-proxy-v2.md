# Local evaluator repair — proxy-v2

Date: 2026-09-01  
Status: `E0 / in progress`

## Why the previous local ranking was unreliable

- `official-shape-v1` selected a hash prefix of windows that concentrated on a few documents and
  used validation only. It also exposed a single fixed test length.
- The local NVFP4 simulator clamped every E4M3 scale below `2^-6` to the minimum normal value;
  E4M3 subnormals down to `2^-9` are representable. This changed most small weight blocks.
- An intermediate rewrite calibrated once per selected case. That is not the observed judge call
  graph: official v86 telemetry has 168 Weight calibration calls (24 layers × 7 roles), 24
  Attention calibration calls, then dynamic calls over the selected cases. Per-case calibration
  gives output-aware candidates an unintended oracle.
- An equal-weight display of case sums was incorrectly treated as an official-score proxy, even
  though official weight revisions and hidden data were not the same.

## Implemented changes

`evaluator/official_eval.py` now uses `proxy-v2`:

- train calibration plus alternating validation/test holdout documents;
- five calibration lengths `[10,128,512,1024,1024]` and a deterministic variable-length test schedule;
- default full Cartesian enumeration of captured W/A tensors (24 layers × 7 Linear roles × every
  holdout window, plus every Attention layer × every holdout window); optional case limits are
  smoke-only and do not define a ranking panel;
- shared state lifetime: all layer/role Weight states use the first two Linear folds, all layer
  Attention states use the five folds, and dynamic calls are one per selected case;
- unweighted `overall_mean`, per-role/layer diagnostics, and an explicit same-cohort
  `trend_diagnostics` pairwise audit. No Linear:Attention ratio or official score is fitted into a
  candidate score.

`evaluator/nvfp4_sim.py` now preserves the E4M3 subnormal range in its round-up operation. The
input codec is recorded as `e4m3-subnormal-ceil-v1`; old v1 caches are rejected rather than silently
reused.

The cache validator also checks every role's real input width against `W.shape[1]` (the Qwen
`proj/down_proj` role is 4864-wide, while the hidden-size roles are 896-wide), all layer/sample
counts, Q/K/V widths, and duplicate holdout windows. The 10.9 GB proxy-v2 cache passed this check;
the initial validator failure was corrected before any score was accepted.

## Evidence already observed

Using the first proxy-v2 cache before the shared-state correction, 25 Linear + 20 Attention cases
completed for v86 and v138. v86 was `0.5197189 / 0.7251850`; v138 was `0.6164252 / 0.7421696`.
Thus v138 still led locally while its user-confirmed official score was below v86 (`15715` vs
`16744`). This is recorded as an `inversion_detected` diagnostic, not hidden by a fitted weight.
That run was a development sample and is not a final proxy result. The final protocol now defaults
to every captured W/A tensor; a smoke limit is allowed only for API/format debugging. The next
full run must record the shared-state call counts (`168`, `24`, then one dynamic call per real
Linear/Attention case) and compare candidates only on that identical full cache.

No local seconds in this log are converted to official platform time, and no official result is
changed by the proxy rewrite.
