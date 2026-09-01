# HiF4 optimization workspace (official-aligned evaluator)

> **Latest official update (2026-09-01):** the user reports that a new Linear framework has
> raised the highest official score to **17816**, 1,072 points above the repository's previous
> v86 high of 16744. The framework combines equivalent Smooth/Permutation/block-Hadamard
> transforms, Weight and Activation GPTQ, and hierarchical HiF4 quadratic refinement. Its official
> runtime, version identifier, and source SHA have not yet been provided, so no `<300s` claim is
> inferred. The active plan now analyzes and extends this framework instead of tuning v140.

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
`16517 / 252.563s`, and v86 **`16744 / 222.7s`** (revised weights). Before the newly reported
17816 Linear framework, v86 was the repository's best revised-weight official result: `+227`
score and `29.863s` faster than v84.
The later v128 fixed-attn-budget candidate was confirmed by the user to time out on the
official evaluator (`>300s`; no official score returned).
The v129 fixed-attn-budget-sweep1 follow-up was also confirmed to time out (`>300s`).
The v130 output-supervised-weight candidate was likewise confirmed to time out (`>300s`),
despite a local API total of `295.437s`.  This is evidence that local seconds do not map
monotonically to the official runtime; the v86 static Attention path remains the time-safe
reference.
The v131 Q(W)-Gram follow-up was also confirmed to time out (`>300s`).  v129-v131 share
the same high-cost Attention family, so the timeout does not isolate the Linear Q(W)-Gram
change as the cause.
The v134 block output-supervised activation cross terms raise the local Linear mean to
`0.5073195`, but v130's official timeout shows that local API seconds are not a safe
runtime proxy. The root file still contains the v140 experiment, but its local Linear gain over
v138 is only `0.000035`; the user reported **`15838 / 207s`** officially. It passed the time
limit but is rejected because it is below v86, and is no longer treated as the best candidate.
Once synchronized, the 17816 source becomes the next Linear implementation baseline.
The user has now reported v138's official result as **`15715 / 208s` (pass)**; its local
proxy values remain separate from that official result. v86 remains the best timed official
source currently present in this repository; the reported 17816 source and runtime are not yet present.
The user has now also reported v139's official result as **`15716 / 202s` (pass)**. Both v138
and v139 remain roughly 1,029 points below v86, so the v138-v145 lineage is closed. The existing
fixed-A comparison v138/v139/v140 gives official deltas `+1/+123`, while v84/v86 keep Linear
identical and gain `+227` from Attention. Therefore later Linear work freezes v86 Attention;
public local means cannot be converted into official Linear/Attention weights. See
[`the attribution record`](docs/evaluation-attribution-2026-09-01.md).
The v147 attribution control now combines v140 Linear with v86 Attention: local means are
`0.5073546371 / 0.7196960689`, API time is `222.227s`, and no official score is inferred.
The v148 fixed A3 alternating round raised local Linear to `0.5097287173` but took `369.038s`
API time, so it is rejected; the next work is complexity reuse, not parameter tuning.
The v141-v145 rank-4 selected-column BDLR trials (anchor freeze, dynamic-only, and two damping
values) returned Linear means `0.281760/0.282559/0.361154/0.506418/0.506256`, all below v140;
that direction is closed. Their source snapshots were deleted to keep the archive compact; per-run
JSON and execution logs remain. The new plan attributes the 17816 framework first, then extends it
with block-Schur GPTQ, low-rank-plus-block-diagonal dynamic Hessians, and joint two-sided residual
optimization.

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
