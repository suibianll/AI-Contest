# HiF4 solutions archive

> **Rule correction (2026-09-05):** official submissions are unlimited. Historical
> `x/10`, “remaining slot”, and quota wording is obsolete; see
> [`stale-information-inventory-2026-09-05.md`](../docs/stale-information-inventory-2026-09-05.md).

> **Official update (2026-09-04):** v183 scored **17598 / 279.7s**, tying v182 while taking
> `6.7s` longer, so it is REJECTED under its pre-registered rule and the attention block-smooth
> coverage family is closed. v182 archive SHA `F3E39E99...A438` remains the score parent at
> **17598 / 273s**. The independent side
> parents remain `P_L=v166（4590/226s）` and `P_A=v168（14005/210s）`; v182 is not an isolated
> Linear measurement. The user-reported leaderboard best is **21765 / 290s**, leaving a
> **4167-point** gap. v180 remains the time-budget parent because it gives up only 1 point for 31s.

`solutions/` contains immutable `solution.py` snapshots. The active code is only the repository
root [`solution.py`](../solution.py). Every snapshot below was submitted or retained as an
official-result candidate; official numbers are historical facts and are not replaced by local
proxy scores.

## Canonical re-evaluation

Use [`evaluator/eval.py`](../evaluator/eval.py), never the retired `real_model_suite.py`.
The current evaluator is `eval-v3`: it reuses the fixed `proxy-v2` dense cache, splits the panel
into six shards, caches calibration artifacts, and emits per-case evidence. `official_eval.py`
is intentionally left unchanged as a `proxy-v2` compatibility/reference backend; the older v1
archive is immutable historical evidence only:

```powershell
.venv\Scripts\python.exe -u evaluator\eval.py --official-audit `
  --cohort new-weight --scenario both --shards 0,1,2,3,4,5 `
  --cache artifacts\official_eval\cache\qwen2.5-0.5b-proxy-v2.pt `
  --calibration-cache-mode auto --algorithm-device cuda `
  --output-dir artifacts\proxy_v3\official-audit
```

The generated audit is diagnostic: official score/time remain independent observations and no
local-to-official score conversion is fitted. Use `--cohort old-weight` explicitly for historical
rows; a cache-cohort mismatch is reported rather than silently mixed.

The protocol fixes Qwen2.5-0.5B, the five Attention calibration lengths
`[10,128,512,1024,1024]`, validation/test holdout windows, independent HiF4 validation, and the
public relative-MSE case score. The default panel is a deterministic stratified real-W/A panel:
168 Linear cases cover every layer/role once and 120 Attention cases cover every layer at each of
the five official lengths. `--full-cases` expands all captured windows for stress; case limits are
still smoke-only. Calibration follows the judge graph: 168 shared layer/role Weight states and 24
shared Attention states, followed by one dynamic call per selected case. The authoritative local
fields are `linear_mean`, `attention_mean`, and the unweighted `overall_mean`; no Linear:Attention
ratio or official-score fit is applied. Each result also contains evaluator-only error-source
controls in `decomposition` and per-case `case_scores`: Linear W/A four-arm output MSE, Attention
Q/K/V/QK controls, and logits/softmax metrics. These controls reuse candidate outputs and do not
change API call counts; use `--no-decomposition` only for a fast smoke run. Local seconds are
same-machine A/B data, not an official-time conversion; `trend_diagnostics` reports known same-
cohort ordering inversions without fitting them.

The eval-v3 audit deliberately uses six balanced shards (336 Linear + 48 Attention cases per
version) so a dense cache is loaded once and calibration artifacts can be reused. Those shard
statistics are diagnostic and must not be mixed with the compatibility backend's 168+120 default
panel.

### Current best and scope rule (2026-09-04)

The highest official score bound to a repository source is **v182: 17598 / 273s**. Its exact
complete parent v180 is **17597 / 242s**; both are Pareto-optimal, so v182 is the score parent and
v180 remains the time-budget parent. v183 `17598/279.7s` is dominated by v182 and rejected. The
independent side parents remain v166 and v168. The
user-reported leaderboard best **21765 / 290s** is 4167 points above v182 and has no synchronized
source or configuration, so it is a target only. v147 is **16579 / 211 s**
(time pass but below v86, rejected); v140 is **15838 / 207 s** (rejected). The pre-A3 parent effect
control is local-only (`Linear=0.588023229`, `Attention=0.757433277`, API `202.317 s`) and is not
an official score.

Do not rank the following together: `default-panel` is the only local proxy-ranking scope;
`effect-panel`/`paired-json-replay` are parent-child mechanism diagnostics, `full-stress` is a
stress check, `smoke-prefix` is interface-only, and GPT-2/hif4/old `official-shape-v1` are
cross-structure or historical probes. Every new JSON records this in `evaluation_scope`; see
[`artifact scope contract`](../artifacts/official_eval/README.md).

The completed 2026-09-01 **historical v1** archive run is in
[`artifacts/official_eval/legacy-v1/archive-official-shape-v1.json`](../artifacts/official_eval/legacy-v1/archive-official-shape-v1.json)
and [`logs/official_eval/archive-official-shape-v1.md`](../logs/official_eval/archive-official-shape-v1.md).
It is immutable historical evidence; do not rank with it.
Among candidates whose functions returned, v121 is the local maximum
(`linear_mean=0.472197763`, `attention_mean=0.833617251`, equal-weight display `28477.289`),
but its API time is `3404.369 s` and its official outcome is timeout. The highest official-pass
candidate under the local API proxy is v084 for Attention (`0.718106989`); v024 is highest for
Linear (`0.450074554`). v002 is recorded as a real local CUDA/CPU device-mix error rather than
silently assigned a score.

## Versions with official outcomes

> 代际声明：v001–v074 为**旧权重**官方分数（与当前评测集不可换算，仅历史证据）；
> v084 起为**新权重**官方分数（当前口径）。v074 另有当前评测集回传 `14561 / 188.9s`
> （2026-09-02），旧权重 `22750` 已失效。

| Version | Source directory | Official score | Official time | Outcome 
|---|---|---:|---:|---
| v001 | `20260826_v001_current-baseline_score10250_time127s` | 10250 | 127 s | pass 
| v002 | `20260826_v002_youxilee-hif4_score15000plus_timeNA` | 15313 | 137 s | pass 
| v013 | `20260827_v013_c10-wide-activation-quadratic_score15799_time144s` | 15799 | 144 s | pass 
| v024 | `20260827_v024_c21-gated-exact-cross-selection_score16043_time174s` | 16043 | 173.8 s | pass 
| v025 | `20260827_v025_c21c-compliance-baseline` | 14437 | 166.6 s | pass 
| v030 | `20260828_v030_c38-beam2-fullcov-official14092_time170.6s` | 14092 | 170.57 s | pass 
| v031 | `20260828_v031_c39-fw-official21864_time161.3s` | 21864 | 161.3 s | pass 
| v032 | `20260828_v032_c40-robust-blockldlq_official-score14432_time216.667s` | 14432 | 216.667 s | pass 
| v034 | `20260829_v034_c41b-mha-k-center_scoreNA_timeNA` | 21864 | 159.4 s | pass 
| v051 | `20260829_v051_c47b-grouping-threshold005_scoreNA_timeNA` | 22451 | 234 s | pass 
| v066 | `20260829_v066_c66-activation-ratio100_scoreNA_timeNA` | 22557 | 217.2 s | pass 
| v072 | `20260829_v072_c74-jdrq-hierarchy_scoreNA_timeNA` | 22662 | 226 s | pass 
| v074 | `20260829_v074_c75-rowwise-jdrq_scoreNA_timeNA` | 22750（旧权重）→ **14561**（当前评测集回传，2026-09-02） | 239.387 s → 188.9 s | pass（**非安全基线**，低于 v84/v86） 
| v084 | `20260830_v084_c84-gram64-sweep5_scoreNA_timeNA` | 16517 | 252.563 s | pass (revised weights) 
| v086 | `20260830_v086_c86-attn-block-final_scoreNA_timeNA` | **16744** | **222.7 s** | **pass (revised weights, new best)** 
| v098 | `20260830_v098_b1-gqrb-margin-active_score293.793700_time406s` | — | >300 s | timeout 
| v100 | `20260830_v100_b2-pawv-diagonly-active_score293.797301_time392s` | — | >300 s | Attention WA / timeout 
| v107 | `20260830_v107_l3-global-lrh-precision-parent_score295.157057_time481s` | — | — | Attention WA 
| v121 | `20260831_v121_c1b-structured-refresh2-accepted_score295.811281_time2180s` | — | >300 s | timeout 
| v128 | `20260901_v128_fixed-attn-budget_timeout` | — | >300 s | **timeout (official, user confirmed)** 
| v129 | `20260901_v129_fixed-attn-budget-sweep1_timeout` | — | >300 s | **timeout (official, user confirmed)** 
| v130 | `20260901_v130_output-weight_timeout` | — | >300 s | **timeout (official, user confirmed)** 
| v131 | `20260901_v131_output-weight-qwgram_timeout` | — | >300 s | **timeout (official, user confirmed)** 
| v138 | `20260901_v138_attention-static-v86-budget_scoreNA_timeNA` | **15715** | **208 s** | **pass (official, user reported)** 
| v139 | `20260901_v139_linear-output-aware-gain_scoreNA_timeNA` | **15716** | **202 s** | **pass (official, user reported)** 
| v140 | `20260901_v140_linear-roab-pair_rejected` | **15838** | **207 s** | **pass, but rejected: below v86 and 17816** 
| v147 | `20260901_v147_v86-attention-v140-linear_rejected` | **16579** | **211 s** | **pass, but rejected: 165 points below v86; submitted SHA unconfirmed** 
| v155 | `20260902_v155_l5a-permutation-stability_rejected` | **16581** | **208.5 s** | **pass, but rejected: 163 points below v86** 
| v156 | `20260902_v156_l4-weight-decoupled_rejected` | **16580** | **204.3 s** | **pass, but rejected: 164 points below v86** 
| v157 | `20260902_v157_v86-roab-only_rejected` | **16729** | **218.96 s** | **pass, but rejected: 15 points below v86** 
| v158 | `20260902_v158_v86-attention-matrix-smooth_retained` | **16861** | **223 s** | **pass; retained, +117 vs v86** 
| v159 | `20260902_v159_linear-gptq17816_v158-attention_score17532_timeNA` | **17532** | — | **official score reported; runtime unknown** 
| v160 | `20260903_v160_v159-linear-l1batch_v158-attn-a2_scoreNA_timeNA` | **17532** | **232 s** | **pass; score no-op vs v159, source/time-complete experiment parent** 
| v161 | `20260903_v161_v160-attn-s1-qk-gram-refine_scoreNA_timeout` | — | >300 s | **timeout (official, user confirmed); local funnel passed (Qwen default 120 paired +0.0525, 106+/14−; GPT-2 +0.0678 same sign; D1 satisfied locally) but per-call dynamic refinement exceeds the official runtime budget — per-call family closed** 
| v162 | `20260903_v162_standard-baseline-both_scoreNA_timeNA` | **1001** | **146 s** | **pass; calibration anchor measured — official non-zero base score or official STD differs from the local reference codec (local means both exactly 0.0); also establishes the ~146 s official harness time floor** 
| v163 | `20260903_v163_v160-linear_standard-attn_scoreNA_timeNA` | **4587** | **202 s** | **pass; official Linear-side contribution Δ_L = 4587−1001 = 3586, local linear mean 0.633526 (bit-identical to v160, 168 cases), attention mean 0.0; time 202s vs predicted ~186s, within margin** |
| v164 | `20260903_v164_standard-linear_v160-attn_scoreNA_timeNA` | **13945** | **204 s** | **pass; official Attention-side contribution Δ_A = 13945−1001 = 12944; together with v163, endpoint additivity predicts v160 within 1 point**
| v165 | `20260903_v165_standard-linear_v161-attn_scoreNA_timeout` | — | >300 s | **timeout (official, user confirmed); standard Linear is bit-identical to v164, so the result isolates the v161 Cross-Gram64 per-call Attention path as over budget; no score means no Attention accuracy ratio is computed**
| v166 | `20260903_v166_rank1-linear-residual_standard-attn_scoreNA_timeNA` | **4590** | **226 s** | **pass; official +3 over v163 (4587/202s), retained as the new Linear parent side P_L (C_L = 4590−1001 = 3589, G_L = +3, 226s < 300s); rank-1 residual redistribution Linear (fold-median top-2 base-codec residual directions, c=1/4, exact product preservation, single-encode design) + standard Attention (mean 0.0, 0/0/120 vs standard); local linear default 0.636590 vs parent 0.633526 (paired +0.003064, 78+/90−, proj +0.0251), API 282.8s (1.24×)**
| v167 | `20260903_v167_standard-linear_lowrank-gram-attn_rejected` | — | — | **rejected (local, pre-official); side-isolation plan 7.2 low-rank Gram codebook — the designated v165-timeout recovery path. Implementation proven correct (lam=0 ablation is bit-identical to parent, 0.797462). Root cause: the real QK cross-Gram is high-rank (top-2 off-diag ~7% eigen mass), so rank-2 coupling-motivated bumps destroy deep sentinels (L15 0.735→−0.54, L23 0.606→−0.05) under both median and mean fold aggregation, while diagonal-only is a mathematical no-op for nearest-level encodings. No rank neighborhood scan per plan 5; Attention-internal mechanisms exhausted**
| v168 | `20260903_v168_standard-linear_logit-gain-attn_scoreNA_timeNA` | **14005** | **210 s** | **pass, RETAINED; new Attention parent P_A. Expansion plan A1: per-KV-head multiplicative logit gain folded into the Q/K multiplier path, zero dynamic additions. step_gain +60 over v164 (Attention ratio 0.46 percent), a small positive gain with no local-proxy signal (local Qwen default mean -0.00088, GPT-2 +0.0024); time +6s over v164. Corrected same-day: initially reported 17248/237s in error. Combined-side prediction with v166: 4590 + 14005 - 1001 = 17594 (+62 over v160)** |
| v180 | `20260904_v180_a1-asym-fold-attn_scoreNA_timeNA` | **17597** | **242 s** | **pass, RETAINED as new full official parent; post-official plan D1 A1 Q/K asymmetric fold (alpha=0.3, exponent-sum keeps logits=gamma, only Q/K dynamic-range reallocation; alpha=0 bit-identical to v175). step_gain +3 vs v175 17594; time −3s is recorded but not claimed as a stable speedup because D1 adds no online operator. Compact 4 paired v175 mean +0.000088 (3+/1−); default 120 paired v168 mean +0.000356 (69+/51−, win 0.575), QK interaction +0.01106; GPT-2 −0.008984 model-specific-risk; opt-125m vs v160 −0.000208 (28+/32−) but D1 increment vs v175 +0.00118 (win 0.467, weak positive, not fully agreeing with GPT-2 sign). Gap to 21765 is 4168** |
| v181 | `20260904_v181_a1-qhead-gain-attn_rejected` | — | — | **rejected (local pre-research, clearly negative); post-official plan D2 per-Q-head logits gain control (each Q head own multiplicative gain on top of A1, K per-KV-head shared — breaks GQA group consistency). Clean D2 (D1 fold OFF, A1 symmetric parent): default 120 paired v168 mean −0.002746, median −0.000086, 54+/66−, median MSE ratio 1.000333; D1+D2 stacked was also negative (mean −0.002019, 60/60). Confirms A1 group-consistent structure is load-bearing; D2 family closed and was not submitted** |
| v182 | `20260904_v182_rank2-linear_v180-attn_scoreNA_timeNA` | **17598** | **273 s** | **pass, RETAINED as new full official parent; post-v180 plan L-R2 fused rank-2 residual redistribution (v166 rank-1 → rank-2 U=[u1,u2]/V=[v1,v2], V^T U≈0, Woodbury R^-1=I-UV^T, continuous product exactly preserved; Attention v180 bit-identical). step_gain +1 vs v180 17597; rank-2 family saturated (0<G_L≤20 → close rank-3/coef scan). Hard checks: reachability all 1, vtu_cross_max ~1e-8, continuous-domain rel err 3.96e-7 (float32). Local paired v180: Qwen compact −0.000093, Qwen default +0.000020, GPT-2 +0.001171, OPT +0.025632 — non-negative, no model-specific-risk. Time 273s (+31) within 300s but margin 27s. Gap to 21765 is 4167** |
| v183 | `20260904_v183_attn-bsm-full-refine_rejected` | **17598** | **279.7 s** | **rejected (official 2026-09-04); direction-1 coverage diag product: v182 + attention block-smooth search refine coverage 0.50→1.00 / blocks 131072 (2 constants only, calibration-side, zero online additions; Linear v182 bit-identical). step_gain 0 vs v182 17598 (score tie, no improvement) per pre-registered rule S≤17598 → REJECTED, coverage family closed. Time 279.7s < 300s (+6.7s calibration refine cost, not timeout). Local: Qwen default +0.000511, GPT-2 −0.005, OPT no_effect. Official parent remains v182** |
| v185 | `20260904_v185_cleanroom-robust-operator_rejected` | **8446** | **165 s** | **official REJECTED; clean-room six-API implementation with analytic Linear diagonal transform and low-DOF Attention K-center/QK-balance/logit-gain/+4 gates. Legal and fast, but official −9153 vs v186 confirms severe algorithm underfit rather than timeout. Balance/gamma/refine neighborhood closed** |
| v184 | `20260904_v184_attn-plus4-gate_timeout` | — | **>300 s** | **timeout (official 2026-09-04); dual-window full-calibration +4 gate (each layer calibrated twice: 4-code + 5-code windows, deployed-MSE gate selects). Time-model attribution: dual-window x2 calibration +36.6s official (0.694 x 52.7s local A_calib) → predicted 309.6s from v182 parent 273s, matches actual >300s timeout. Root cause is the dual-window architecture, NOT the 5-code window itself (single-window 5-code calibration runs at 4-code speed: 66.0s vs 66.1s local). Local: Qwen +0.006580 (L11 +0.158 recovered via gate-decision flip), GPT-2 no_effect 12/12 rejected, OPT −0.0002. Timeout per pre-registered rule; single-window +4 variant (the probe, zero calibration cost, Qwen +0.010386) is the time-safe restructure of the same mechanism** |
| v186 | `20260904_v186_attn-plus4-single-window_scoreNA_timeNA` | **17599** | **272 s** | **pass, RETAINED as new full official parent; oracle-decomposition minimal product: v182 + 1-line `_DYNAMIC_OFFSETS (-1,1,2,3)→(-1,1,2,3,4)` (add single +4 E6M2 code to online Q/K/V scale window; hill-climb edge extension cannot reach it across binades). step_gain +1 vs v182 17598; time −1s (time-model predicted 274.0s, actual 272s, within MAE 10.1s — calibration-neutral prediction validated). Local Δmean +0.010344 (largest post-A1 signal, 29x D1) → official +1: reconfirms local mean does not convert to official points but sign gates (Δmean>0, L1=0.0155<0.02) were zero-error. Family officially positive; no code-neighborhood scan (+5/-2 etc.). Gap to 21765 is 4166; time margin 28s** |
| v187 | `20260904_v187_attn-jacobian-sensitivity_research-retained` | **9167** | **169 s** | **official positive / RESEARCH RETAINED; v185 clean-room + analytic final-Attention Jacobian importance for Q/K, KV-group shared and leave-one-fold-out gated. Official +721/+4s vs v185 confirms transfer. 7/24 layers active; local Δmean +0.015187, L1 0.016199. Still −8432 vs v186, so not a full parent; root unchanged** |
| v188 | `20260904_v188_attn-jacobian-port_rejected` | **17595** | **268 s** | **rejected (official 2026-09-04); v186 + v187 Jacobian sensitivity importance ported as final calibration step on the fully-transformed Q/K coordinates (causal/non-causal 0.5, cross-fold median, log shrink 0.25, clamp [0.5,2], LOO deployed-MSE gate; v187 pre-registered constants, no neighborhood scan). step_gain −4 vs v186 17599; time 268s (model predicted 274s, within MAE). Local default 120 vs v186: Δmean +0.000426, L1 0.001114, 6+/4−/110=; gate accepted only 2/24 layers (L12/L22 — the pair-transform-free layers; pair-smooth output-fitted importance wins elsewhere). First sign-gate miss on a near-zero local signal (110/120 cases unchanged): the gate blocks large losses (−165~−1164) but does not guarantee non-negative official deltas for near-zero signals; official ±1~4 is the effective noise band (single-point gains v182/v186 were +1/+1/+3). Jacobian port family closed; root rolled back to v186** |
| v169 | `20260903_v169_standard-linear_v-bias-attn_rejected` | — | — | **rejected (local, clearly negative); expansion plan A2 V output-bias centroid: local Qwen -0.0093 (21+/99-) and GPT-2 0/4 all-negative - final classification per user 'reject clearly-negative optimizations'** |
| v170 | `20260903_v170_standard-linear_fixed-offset-attn_rejected` | — | — | **rejected (local, clearly negative); expansion plan A3 static fixed-offset compile: Qwen -0.0506 (9+/111-) and GPT-2 -0.0551 (1+/3-) - final classification per user** |
| v171 | `20260903_v171_standard-linear_moment-threshold-attn_rejected` | **13657** | **214 s** | **rejected (official 2026-09-04); expansion plan A4 moment-matched mantissa rounding threshold. step_gain −348 vs v168 (14005), Attention ratio −2.69%. Time 214s < 300s; negative from algorithm not timeout. A4 family closed** |
| v172 | `20260903_v172_babai-weight-decode_rejected` | — | — | **rejected (local, clearly negative); expansion plan L2 HiF4 hierarchical Babai decode: compact -0.0483, 0+/48- (zero positive cases) - final classification per user** |
| v173 | `20260903_v173_trellis-vq-weight-decode_rejected` | — | — | **rejected (local, clearly negative); expansion plan L3 fixed-width Trellis/VQ: compact -0.0229, 1+/47- (single positive case) - final classification per user** |
| v174 | `20260903_v174_kronecker-cat_standard-attn_rejected` | **4508** | **190 s** | **rejected (official 2026-09-04); expansion plan L4 Kronecker-compressed analytic CAT. step_gain −82 vs v166 (4590), Linear ratio −2.29%. Time 190s < 300s; L4 family closed** |
| v175 | `20260903_v175_rank1-linear_logit-gain-attn_scoreNA_timeNA` | **17594** | **245 s** | **pass, RETAINED as new full official parent; combination (plan 13) v166 rank-1 Linear + v168 A1 logit-gain Attention. interaction = 17594−4590−14005+1001 = 0 (exact additivity confirmed on official total). S_pred=17594 (+62 over v160, 4171 from leaderboard 21765). Time 245s < 300s** |
| v176 | `20260903_v176_k-outlier-eq-attn_rejected` | **13964** | **205 s** | **rejected (official 2026-09-04); next-stage plan C1 K-side static outlier-channel equalization. step_gain −41 vs v168 (14005), Attention ratio −0.32%. Time 205s < 300s; C1 family closed (consistent with local default −0.004450, GPT-2 −0.002753, opt −0.021851)** |
| v177 | `20260904_v177_c2-group-logit-gain_rejected` | — | — | **rejected (local pre-research, clearly negative); plan C2 A1-fine-grained per-(KV-head, 8-channel-group) logits gain (closed-form 8-param LS per §2b), on P_A=v168. Compact 4 paired v168 mean −0.006858 (1+/3−); default 120 mean −0.006643 (41+/79−, win 0.342), QK interaction −0.0935. B=8 groups negative on v168 branch; not submitted** |
| v178 | `20260904_v178_c2-on-c1-pre-research_rejected` | — | — | **rejected (local pre-research, clearly negative); C2 C1-branch preview (per §2b, C2 on v176 = C1 semantic + group gain). Compact 4 paired v176 mean −0.009194 (1+/3−); default 120 paired v168 mean −0.008610 (47+/73−, win 0.392) — worse than v177. C2 family closed on both branches (v168/C1); not submitted** |
| v179 | `20260904_v179_c3-rand8-ortho-attn_rejected` | — | — | **rejected (local pre-research, clearly negative); plan C3 fixed 8×8 random orthogonal rotation (QuaRot/TurboQuant control, per-KV-head fixed-seed QR, QK^T inner-product-invariant). Compact 4 paired v168 mean −0.229273 (1+/3−), median MSE ratio 1.512 — largest negative among tested mechanisms, matching Longhorn Qwen GQA rotation delocalize evidence. C3 family closed; not submitted** |
| v169 | `20260903_v169_standard-linear_v-bias-attn_rejected` | — | — | **rejected (local, pre-official); expansion plan A2 V output-bias centroid: per-KV-head b = 0.5 * fold-median of mean(O_ref - O_parent), added to V right after the NVFP4 decode. Controls perfect (Q/K states bit-identical). Rejected on three independent grounds: mechanism-level premise absent (output-bias correction only -1.7%..+0.6%, parent output bias is Q/K-dominated), Qwen default -0.0093 (21+/99-, V-only -0.0154), GPT-2 0/4 all-negative (-0.0172) triggering the plan 12-step-9 cross-model structural-reversal block. It was not submitted** |
| v170 | `20260903_v170_standard-linear_fixed-offset-attn_rejected` | — | — | **rejected (local, pre-official); expansion plan A3 static compile of the dynamic scale search: fixed E6M2 offset + exact hierarchy, Q->K->V greedy on real attention output MSE. Winners 11/12 = 0 (standard scale already output-optimal). Cross-model structural reversal, far stronger than v169: Qwen default -0.0506 (9+/111-, all lengths negative) and GPT-2 -0.055 (1+/3-, k_only -0.093). The dynamic refine is a load-bearing part of the official 12944 Attention contribution. A1 multiplier bit-identical (control clean). It was not submitted; A3 closed, next A4**

## 2026-09-01 official-shape-v1 local candidates

These are local reproductions only; no official score/time is inferred from them.  Their
directories follow the same immutable naming rule as the historical archive:
`YYYYMMDD_vNNN_<description>_scoreNA_timeNA`.

| Version | Source directory | Linear mean | Attention mean | API total | Decision 
|---|---|---:|---:|---:|---
| v086 (idle rerun) | `20260830_v086_c86-attn-block-final_scoreNA_timeNA` | 0.406668 | 0.719696 | 299.302 s | clean rerun; official 16744/222.7 s pass 
| v128 | `20260901_v128_fixed-attn-budget_timeout` | 0.465655 | 0.837789 | 310.732 s | **official timeout (user confirmed)** 
| v129 | `20260901_v129_fixed-attn-budget-sweep1_timeout` | 0.465655 | 0.836579 | 248.363 s | **official timeout (user confirmed)** 
| v130 | `20260901_v130_output-weight_timeout` | 0.471837 | 0.836579 | 295.437 s | **official timeout (user confirmed); Attention-time risk** 
| v131 | `20260901_v131_output-weight-qwgram_timeout` | 0.473131 | 0.836579 | 294.835 s | **official timeout; high-cost Attention family** 
| v132 | `20260901_v132_output-weight-qwgram-dynsweep2_scoreNA_timeNA` | 0.473131 | 0.834256 | 290.936 s | historical parent; 2 idle runs API<300 
| v133 | `20260901_v133_output-weight-qwgram-gain_scoreNA_timeNA` | 0.483610 | 0.834256 | 287.941 s | historical parent 
| v134 | `20260901_v134_linear-output-activation-cross64_scoreNA_timeNA` | 0.507320 | 0.834256 | 289.042/289.832 s | Linear precision parent; Attention-time risk 
| v135–v137 | three directories explicitly suffixed `_rejected` | 0.500132–0.507163 | 0.834256 | 287.816–296.755 s | **rejected Jacobi/sweep variants** 
| v138 | `20260901_v138_attention-static-v86-budget_scoreNA_timeNA` | **0.507320** | 0.715942 | **192.996/187.935 s** | **official 15715/208 s pass; time parent** 
| v139 | `20260901_v139_linear-output-aware-gain_scoreNA_timeNA` | 0.507278 | 0.715942 | 193.389 s | **official 15716/202 s pass; retained official-result archive** 
| v140 | `20260901_v140_linear-roab-pair_rejected` | 0.507355 | 0.715942 | 205.365 s | **rejected; official 15838/207 s, local-only gain `+0.000035`** 
| v141–v145 (BDLR family) | — (source snapshots deleted; logs/artifacts retained) | 0.281760–0.506256 | 0.715942 | 204.681–211.460 s | **rejected family; selected-column BDLR closed** 
| v147 | `20260901_v147_v86-attention-v140-linear_rejected` | **0.507355 / 0.510050†** | **0.719696** | **222.227 / 300.351 s†** | **official 16579/211s; rejected below v86; submitted SHA unconfirmed** 
| v148 | `20260901_v148_joint-wa-v86-attention-v140-linear_rejected` | **0.509729** | **0.719696** | **369.038 s** | **rejected; A3 precision gain but local time over 300 s** 
| v151 | `20260902_v151_proj-roab-off_rejected` | **0.582528 (targeted)** | **0.942927 (targeted)** | **193.213 s** | **rejected; Qwen role panel no-op, GPT-2 proj-only gain** 
| v152 | `20260902_v152_fc-cat-off_rejected` | **0.583139 / 0.542553 (14/56-case)** | **0.942927** | **199.578/200.432 s** | **rejected; small mixed-sign fc gain** 
| v153 | `20260902_v153_fc-decoupled-activation_rejected` | **0.568754 (targeted)** | **0.942927** | **197.656 s** | **rejected; direct s_q assignment regresses fc** 
| v154 | `20260902_v154_fc-decoupled-scale-fit_rejected` | **0.568754 (targeted)** | **0.942927** | **198.098 s** | **rejected; fitted s_d is a no-op after v153** 
| v155 | `20260902_v155_l5a-permutation-stability_rejected` | **0.570999 (default)** / 0.588162 (effect) | **0.724735 (default)** / 0.757433 (effect) | **248.121 s (default-equivalent)** / 207.196 s (effect) | **rejected; official 16581/208.5s, 163 points below v86** 
| v156 | `20260902_v156_l4-weight-decoupled_rejected` | 0.588131 (effect; default not run) | 0.757433 (effect) | 203.994 s (effect) | **rejected; official 16580/204.3s, 164 points below v86** 
| v157 | `20260902_v157_v86-roab-only_rejected` | NA (legality smoke only) | NA (frozen field-equality check) | NA | **rejected; official 16729/218.96s, 15 points below v86** 
| v158 | `20260902_v158_v86-attention-matrix-smooth_retained` | 0.448180 (default; frozen) | 0.735752 (default) | 295.069 s (default) | **retained; official 16861/223s, +117 vs v86** 
| v159 | `20260902_v159_linear-gptq17816_v158-attention_score17532_timeNA` | **0.705508 CUDA compact / 0.633526 CUDA Linear default** | frozen v158 | **51.055s compact after exact reuse / 269.435s pre-reuse default API** | **official 17532 binds original SHA; current archive not yet resubmitted** 

† The first v147 values come from the original pre-A3 JSON (SHA `9B3EA5...B656`); the second come
from the later direct-merge A3 JSON (SHA `25C245...9C1B`). The archive was modified in place before
the official result was reported, so neither SHA is claimed as the submitted source without further
evidence. The current archived source SHA is `44E377...2672`.

\* The later temporary gain+adyn2 run reported `365.818 s`, but the machine was concurrently busy;
its timing is excluded from runtime ranking. The persisted v133 archive was rerun idle at `291.275 s`;
the active root file was then rerun directly at `287.941 s`.  v134 adds the
output-supervised activation cross term; its two complete runs are recorded in
[`v134 first JSON`](../artifacts/official_eval/v134-linear-output-activation-cross64-official-shape-v1.json)
and [`v134 idle rerun JSON`](../artifacts/official_eval/v134-linear-output-activation-cross64-rerun2-official-shape-v1.json).

The root file currently contains v140 for audit, but the next implementation baseline is the exact
v86 source. v140 keeps the v138 reduced Attention path and adds the ROAB-P2 reciprocal pair
transform to Linear; its local gain is only `0.000035`, and its official result is `15838 / 207 s`
(below v86), so it is rejected. v138 disables
the per-call Attention Gram refinement and shrinks the static candidate set. Two v138 runs are recorded in
[`v138 first JSON`](../artifacts/official_eval/v138-attention-static-v86-budget-official-shape-v1.json)
and [`v138 idle rerun JSON`](../artifacts/official_eval/v138-attention-static-v86-budget-rerun2-official-shape-v1.json).
The v138 official result was reported as **`15715 / 208 s` (pass)**, v139 as **`15716 / 202 s` (pass)**,
and v140 as **`15838 / 207 s` (pass but rejected as inferior)**. The v140 full run is recorded in
[`v140 JSON`](../artifacts/official_eval/v140-linear-roab-pair-official-shape-v1.json);
it gives Linear `0.5073546371`, unchanged Attention `0.7159419612`, and API `205.365 s`.

The BDLR-JAQ trials v141–v145 are recorded as a rejected family summary. Their local Linear means
were `0.281760`, `0.282559`, `0.361154`, `0.506418`, and `0.506256`; all kept Attention at
`0.715942` and stayed below the local time proxy, but none improved v140. Their source snapshots
were deleted to keep the archive compact; the per-run JSON and execution logs remain as evidence.
The selected-column BDLR direction is closed. The next work starts from v86, builds legal structural
oracles, and then tests null-space shaping and subspace-embedded joint vector rounding.

## External hif4 GPT-2 cross-check

The upstream [youxilee/hif4](https://github.com/youxilee/hif4) `real_data_eval.py` was also run
on the v84/v86/v140/v147 snapshots with one fixed 12-layer GPT-2 configuration
(`amax6`, `seq=128`, `calib=2`, `test=2`, `config=current`). Its Linear/Attention means were
`0.586733/0.4477`, `0.586733/0.4727`, `0.599617/0.4661`, and `0.599617/0.4713`, respectively.
This is a diagnostic only: the script repeats a built-in synthetic text when short and obtains
the standard baseline from candidate-private codec functions, so its ordering does not replace
the canonical `proxy-v2` evaluator or official results. See the full
[`external run log`](../logs/execution/2026-09-01-hif4-external-gpt2-v84-v86-v140-v147.md).

### External role attribution

The same hif4 run gives a more actionable result than the aggregate ordering. Relative to v86,
v140 improves static `q/k/v` by `+0.0409/+0.0900/+0.0085`, is nearly neutral on `o` (`-0.0018`),
but regresses `fc` in all 12 GPT-2 layers (`-0.0452`) and has a mixed `proj` regression
(`-0.0153`, including a `-0.1634` layer). A temporary role-gated ablation raises `proj` from
`.5221` to `.5430` when its ROAB is disabled; disabling fc ROAB is a no-op, while disabling fc
BOAT is harmful (`.5107` to `.4599`). The next Linear work therefore freezes q/k/v/o, tests
proj ROAB-off first, and redesigns fc's expansive encoder/scale while retaining BOAT. This is a
role diagnostic, not an official-score claim; the full evidence and protocol caveats are in
[`role attribution log`](../logs/execution/2026-09-01-hif4-external-role-attribution-v140-v86.md).

The first follow-up control, v151, disabled ROAB only for native `rows < channels` (`proj/down`)
matrices while removing the over-budget A3 pass. On the canonical Qwen `proxy-v2` targeted panel,
all seven static Linear role means were identical to the pre-A3 parent (`0.582528` Linear,
`0.942927` Attention); the external four-layer GPT-2 smoke improved only `proj` (`.5029→.5658`).
It is archived as `REJECTED` and remains a cross-model control rather than a new root parent. See
[`v151 execution log`](../logs/execution/2026-09-02-v151-proj-roab-off.md).

The next fc controls were also kept out of root. v152 disabled only expansive CAT (BOAT retained):
Qwen Linear moved `0.582528→0.583139` on the 14-case smoke but only
`0.542366→0.542553` on the paired 56-case panel, with mixed fc layer signs; external GPT-2 moved
fc `.5658→.5709`. v153 then tried the first L1 decoupled encoder, using BF16 `s_q` directly for
fc code assignment without fitting `s_d`, and regressed Qwen Linear to `0.568754` (fc family
`0.334432`). Both are archived as `REJECTED`; the failure points to the planned closed-form stored
scale fit, not another CAT/ROAB switch. See
[`v152/v153 execution log`](../logs/execution/2026-09-02-v152-v153-fc-followups.md).

v154 added the planned fixed-code `s_d` fit in the deployed `Q(W)` Gram metric, but its Qwen role
means were exactly v153 (`fc=.334432`, Linear `0.568754`). It is archived as `REJECTED`; direct
decoupled-scale variants are paused. L3-D0 then found same-fold teacher margin but an exact
layer-3/fold-128 output regression (`fc_gate=-0.094751`, `fc_up=-0.112680`), so the result is
`margin_exists_but_not_compile_safe`. The batched stability probe found no fixed threshold/LUT
student (held-out precision zero), so direct activation encoder compilation is closed. Its only
positive remnant is v155: a fixed four-quartile pressure interleave behind BOAT with a
fold-disagreement gate. It changes only four default Linear cases, gives paired
`+0.000116536` with no regressions, and leaves controls/Attention unchanged; the strict GPT-2
pair is `−0.000153` (fc `−0.000916`), so this is a Qwen-local coordinate control, not a new
official baseline or parent. The first L2 analytic pair-balance probe remains rejected (`fc` paired
mean `-0.314079`, 16/16 regressions). The next experiment is therefore a single-pass
Weight-decoupled or deployment-Gram block-Schur mechanism branched from pre-A3, not another
permutation/scale/CAT sweep.

v157 was the exact-v86 single-variable ROAB experiment. It starts from the reproducible v86
source and adds only the bounded ROAB-P2 reciprocal 2×2 Linear transform, while keeping v86
Attention field-for-field frozen. The direction comes from the clean fixed-Attention official
increment `v138→v140 = +123`, not from a local ranking. No local model panel was run for v157;
legality, continuous-product/covariance invariants, rejected-candidate parent equality, selected
state propagation and isolated import passed. Its SHA is
`984BF752156187B8892894060A99FE52027E2457F37FC23C11657041B29B86E1`. Its official result is
`16729 / 218.96s`: time passes but accuracy is 15 points below v86, so it is rejected. This proves
the earlier `v138→v140 +123` ROAB increment was context-dependent, not portable to exact v86.
v155 (`16581 / 208.5s`) and v156 (`16580 / 204.3s`) remain rejected as well. The next planned
mechanism is a single-pass block-Schur HiF4-GPTQ branch from exact v86.

\* The earlier v086 local `462.239 s` observation was also concurrent-load affected. The clean
idle rerun is `299.302 s` API / `321.996 s` wall; see
[`v086 idle rerun`](../artifacts/official_eval/v086-idle-rerun-20260901-official-shape-v1.json).

## Recording rules

1. Parameter sweeps stay in one unnumbered workbench and one summary log. Allocate a version only
   for a new mathematical algorithm, an official submission, or one representative failure.
2. Final directory names include `retained`, `rejected`, or `timeout`. Unknown official values
   stay `scoreNA_timeNA`; local JSON values never enter Official fields.
3. `result.md` records parent, one algorithm change, exact command/protocol, data/model revisions,
   both means, API/Wall time, source SHA256, an explicit `Status` (`RETAINED`, `REJECTED`,
   `TIMEOUT`, or `ERROR`), official outcome, and next decision.
4. When the official judge changes weights or limits, start a new protocol label and keep old
   outcomes as history; never mix their absolute scores.
5. Small parameter sweeps are grouped under one experiment log and one summary result. Rejected
   micro-variants are not archived one-by-one; their source snapshots may be deleted after JSON and
   execution evidence is retained. Version identifiers are globally unique.
