# HiF4 optimization workspace (official-aligned evaluator)

The only active local evaluator is [`evaluator/official_eval.py`](evaluator/official_eval.py).
The former `real_model_suite.py` / `sampled-means-v1/v2` reports are retired and must not be
used for ranking or timing decisions.

The evaluator fixes the known public contract: Qwen2.5-0.5B, 24 blocks, 250 Linear cases,
200 Attention cases, the five variable Attention calibration lengths `[10, 128, 512, 1024,
1024]`, independent HiF4 validation, and the public per-case score
`(MSE_STD - MSE_PLAYER) / MSE_STD`. It reports `linear_mean`, `attention_mean`, the case-score
sum, API time, and wall time. The official platform currently requires end-to-end time below
300 seconds; local CUDA seconds are an A/B proxy only and are never converted into an official
score. The latest known official anchors are v74 `22750 / 239.387s` (old weights), v84
`16517 / 252.563s`, and v86 **`16744 / 222.7s`** (revised weights). v86 is the best
revised-weight official result so far: `+227` score and `29.863s` faster than v84.

Capture the pinned public data pack once:

```powershell
.venv\Scripts\python.exe -u evaluator\official_eval.py `
  --cache artifacts\official_eval\cache\qwen2.5-0.5b-official-shape-v1.pt `
  --cache-mode write --capture-device cuda --algorithm-device cuda
```

Re-evaluate every archived official-result version with the same cache and configuration:

```powershell
.venv\Scripts\python.exe -u evaluator\official_eval.py --archive `
  --cache artifacts\official_eval\cache\qwen2.5-0.5b-official-shape-v1.pt `
  --cache-mode read --algorithm-device cuda `
  --output artifacts\official_eval\archive-official-shape-v1.json `
  --report logs\official_eval\archive-official-shape-v1.md
```

See the Chinese [`README.md`](README.md) for the archive table, cleanup rules, active-plan
policy, and full reproducibility details. Cached tensors are several GB and are intentionally
not committed.
