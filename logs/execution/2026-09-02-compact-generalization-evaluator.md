# Compact generalization evaluator

- Date: 2026-09-02
- Status: DONE
- Scope: evaluator-only; `solution.py` unchanged
- Protocol: `proxy-v2`; new scope `linear-only-compact-generalization-panel`

## Why

The previous effect panel reduced dynamic scoring cases but still built 168 Linear and 24
Attention calibration states.  It therefore retained almost all candidate API cost, while the
aggregate Linear mean did not expose tail regressions or cross-holdout instability.

## Change

- Added `--compact-panel`.
- Linear selects depth-spread layers `0/8/15/23`, all seven roles, and two same-length
  validation/test holdouts per layer/role: 28 Weight states and 56 dynamic cases.
- Linear calibration uses train folds 1/2 (lengths 128/512, different documents).
- Attention compact selects four depth-spread layer/window sentinels.
- Pack preparation now runs after scenario selection and encodes only reachable weights,
  calibration tensors, test tensors, and Q/K/V tensors.
- Added Linear gain/MSE-ratio median, quartiles, worst-quartile, min/max and sign counts, grouped
  by role/family/layer/shape/split/length.
- Added validation/test cross-holdout sign agreement, gain gap and paired minimum-gain.
- W-only/A-only/Both/interaction distributions reuse the existing evaluator-only decomposition.

This scope is diagnosis-only: `official_score_equivalent=false` and
`comparable_for_proxy_ranking=false`.  The default all-state panel remains a low-frequency audit.

## Verification

Command:

```powershell
.\.venv\Scripts\python.exe -u evaluator\official_eval.py `
  --solution solution.py --name root-compact-generalization-v2 `
  --cache artifacts\official_eval\cache\qwen2.5-0.5b-proxy-v2.pt `
  --cache-mode read --algorithm-device cuda --linear-only --compact-panel `
  --output artifacts\official_eval\root-compact-generalization-linear-v2.json `
  --report logs\official_eval\root-compact-generalization-linear-v2.md
```

Observed:

- cache load + selected preparation: `8.203s`;
- Weight calibration calls: `28`;
- Activation dynamic calls: `56`;
- candidate API total: `40.408s`;
- candidate wall: `45.438s`;
- total local turnaround: about `53.64s`;
- gain median / worst-quartile mean: `0.553221 / 0.421898`;
- validation/test sign agreement: `28/28` pairs;
- gain-gap median / max: `0.009432 / 0.063375`.

The historical v86 default run used `47.904s` preparation plus `322.895s` candidate wall.  The
two scopes are not score-comparable; this comparison only demonstrates evaluator turnaround and
complexity reduction.

Tests:

```text
python -m py_compile evaluator/official_eval.py
python -m pytest tests/test_official_eval.py -q  -> 28 passed
python -m pytest -q                              -> 40 passed
git diff --check                                 -> passed
```
