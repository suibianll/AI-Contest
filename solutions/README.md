# HiF4 solutions archive

`solutions/` contains immutable `solution.py` snapshots. The active code is only the repository
root [`solution.py`](../solution.py). Every snapshot below was submitted or retained as an
official-result candidate; official numbers are historical facts and are not replaced by local
proxy scores.

## Canonical re-evaluation

Use [`evaluator/official_eval.py`](../evaluator/official_eval.py), never the retired
`real_model_suite.py`:

```powershell
.venv\Scripts\python.exe -u evaluator\official_eval.py --archive `
  --cache artifacts\official_eval\cache\qwen2.5-0.5b-official-shape-v1.pt `
  --cache-mode read --algorithm-device cuda `
  --output artifacts\official_eval\archive-official-shape-v1.json `
  --report logs\official_eval\archive-official-shape-v1.md
```

The protocol fixes Qwen2.5-0.5B, 250 Linear + 200 Attention cases, Attention calibration
lengths `[10,128,512,1024,1024]`, independent HiF4 validation, and the public relative-MSE
case score. The authoritative local table is generated from the JSON above; `linear_mean` and
`attention_mean` are the only local ranking metrics. Local seconds are same-machine A/B data,
not an official-time conversion.

The completed 2026-09-01 archive run is in
[`artifacts/official_eval/archive-official-shape-v1.json`](../artifacts/official_eval/archive-official-shape-v1.json)
and [`logs/official_eval/archive-official-shape-v1.md`](../logs/official_eval/archive-official-shape-v1.md).
Among candidates whose functions returned, v121 is the local maximum
(`linear_mean=0.472197763`, `attention_mean=0.833617251`, equal-weight display `28477.289`),
but its API time is `3404.369 s` and its official outcome is timeout. The highest official-pass
candidate under the local API proxy is v084 for Attention (`0.718106989`); v024 is highest for
Linear (`0.450074554`). v002 is recorded as a real local CUDA/CPU device-mix error rather than
silently assigned a score.

## Versions with official outcomes

| Version | Source directory | Official score | Official time | Outcome |
|---|---|---:|---:|---|
| v001 | `20260826_v001_current-baseline_score10250_time127s` | 10250 | 127 s | pass |
| v002 | `20260826_v002_youxilee-hif4_score15000plus_timeNA` | 15313 | 137 s | pass |
| v013 | `20260827_v013_c10-wide-activation-quadratic_score15799_time144s` | 15799 | 144 s | pass |
| v024 | `20260827_v024_c21-gated-exact-cross-selection_score16043_time174s` | 16043 | 173.8 s | pass |
| v025 | `20260827_v025_c21c-compliance-baseline` | 14437 | 166.6 s | pass |
| v030 | `20260828_v030_c38-beam2-fullcov-official14092_time170.6s` | 14092 | 170.57 s | pass |
| v031 | `20260828_v031_c39-fw-official14613_time159.2s` | 21864 | 161.3 s | pass |
| v032 | `20260828_v032_c40-robust-blockldlq_official-score14432_time216.667s` | 14432 | 216.667 s | pass |
| v034 | `20260829_v034_c41b-mha-k-center_scoreNA_timeNA` | 21864 | 159.4 s | pass |
| v051 | `20260829_v051_c47b-grouping-threshold005_scoreNA_timeNA` | 22451 | 234 s | pass |
| v066 | `20260829_v066_c66-activation-ratio100_scoreNA_timeNA` | 22557 | 217.2 s | pass |
| v072 | `20260829_v072_c74-jdrq-hierarchy_scoreNA_timeNA` | 22662 | 226 s | pass |
| v074 | `20260829_v074_c75-rowwise-jdrq_scoreNA_timeNA` | 22750 | 239.387 s | pass |
| v084 | `20260830_v084_c84-gram64-sweep5_scoreNA_timeNA` | 16517 | 252.563 s | pass (revised weights) |
| v086 | `20260830_v086_c86-attn-block-final_scoreNA_timeNA` | **16744** | **222.7 s** | **pass (revised weights, new best)** |
| v098 | `20260830_v098_b1-gqrb-margin-active_score293.793700_time406s` | — | >300 s | timeout |
| v100 | `20260830_v100_b2-pawv-diagonly-active_score293.797301_time392s` | — | >300 s | Attention WA / timeout |
| v107 | `20260830_v107_l3-global-lrh-precision-parent_score295.157057_time481s` | — | — | Attention WA |
| v121 | `20260831_v121_c1b-structured-refresh2-accepted_score295.811281_time2180s` | — | >300 s | timeout |
| v128 | `20260901_v128_fixed-attn-budget_scoreNA_timeNA` | — | >300 s | **timeout (official, user confirmed)** |
| v129 | `20260901_v129_fixed-attn-budget-sweep1_scoreNA_timeNA` | — | >300 s | **timeout (official, user confirmed)** |
| v130 | `20260901_v130_output-weight_scoreNA_timeNA` | — | >300 s | **timeout (official, user confirmed)** |

## 2026-09-01 official-shape-v1 local candidates

These are local reproductions only; no official score/time is inferred from them.  Their
directories follow the same immutable naming rule as the historical archive:
`YYYYMMDD_vNNN_<description>_scoreNA_timeNA`.

| Version | Source directory | Linear mean | Attention mean | API total | Decision |
|---|---|---:|---:|---:|---|
| v086 (idle rerun) | `20260830_v086_c86-attn-block-final_scoreNA_timeNA` | 0.406668 | 0.719696 | 299.302 s | clean rerun; official 16744/222.7 s pass |
| v128 | `20260901_v128_fixed-attn-budget_scoreNA_timeNA` | 0.465655 | 0.837789 | 310.732 s | **official timeout (user confirmed)** |
| v129 | `20260901_v129_fixed-attn-budget-sweep1_scoreNA_timeNA` | 0.465655 | 0.836579 | 248.363 s | **official timeout (user confirmed)** |
| v130 | `20260901_v130_output-weight_scoreNA_timeNA` | 0.471837 | 0.836579 | 295.437 s | **official timeout (user confirmed); Attention-time risk** |
| v131 | `20260901_v131_output-weight-qwgram_scoreNA_timeNA` | 0.473131 | 0.836579 | 294.835 s | Q(W)-Gram parent |
| v132 | `20260901_v132_output-weight-qwgram-dynsweep2_scoreNA_timeNA` | 0.473131 | 0.834256 | 290.936 s | historical parent; 2 idle runs API<300 |
| v133 | `20260901_v133_output-weight-qwgram-gain_scoreNA_timeNA` | 0.483610 | 0.834256 | 287.941 s | historical parent |
| v134 | `20260901_v134_linear-output-activation-cross64_scoreNA_timeNA` | **0.507320** | **0.834256** | **289.042/289.832 s** | **active root; two identical idle reruns, API<300** |

\* The later temporary gain+adyn2 run reported `365.818 s`, but the machine was concurrently busy;
its timing is excluded from runtime ranking. The persisted v133 archive was rerun idle at `291.275 s`;
the active root file was then rerun directly at `287.941 s`.  v134 adds the
output-supervised activation cross term; its two complete runs are recorded in
[`v134 first JSON`](../artifacts/official_eval/v134-linear-output-activation-cross64-official-shape-v1.json)
and [`v134 idle rerun JSON`](../artifacts/official_eval/v134-linear-output-activation-cross64-rerun2-official-shape-v1.json).

\* The earlier v086 local `462.239 s` observation was also concurrent-load affected. The clean
idle rerun is `299.302 s` API / `321.996 s` wall; see
[`v086 idle rerun`](../artifacts/official_eval/v086-idle-rerun-20260901-official-shape-v1.json).

## Recording rules

1. Allocate a new version directory before changing the active root; never overwrite a snapshot.
2. Unknown official values stay `scoreNA_timeNA`. Local JSON values belong in the local report,
   not in the directory name or Official fields.
3. `result.md` records parent, one algorithm change, exact command/protocol, data/model revisions,
   both means, API/Wall time, source SHA256, official outcome, and next decision.
4. When the official judge changes weights or limits, start a new protocol label and keep old
   outcomes as history; never mix their absolute scores.
