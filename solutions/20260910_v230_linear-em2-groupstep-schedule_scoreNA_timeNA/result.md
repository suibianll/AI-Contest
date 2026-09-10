# v230 — Linear group-major exact-metric descent (L-EM2)

Status: `PENDING` official (no official evaluation run; the user runs batched official
evaluations). Local six-shard result is positive; the local diagnostic tool reports
`reject` on its L1 trend screen, which is not a mechanism failure — see below.

- Parent: v202 Linear + v195 Attention complete root `solution.py`
- Parent SHA256: `56DC805D6E5A3AEF896DB8021045740292735725D688B48E3D4393E55EFCB2BD` (485072 B)
- Candidate SHA256: `0f1af6dbc207ff32b2c6be16987e9c4fe50f3f10747de26782ef52a6f2fab7bc` (505496 B)
- Version note: v229 was taken by the parallel Attention line
  (`attention-amc1-k-mean-recenter`) while this candidate was in flight. v230 is a
  **sibling** of v229 — both are parented on the same root `56dc805d`.
- Plan: [`2026-09-10-linear-groupstep-schedule-plan.md`](../../docs/superpowers/plans/2026-09-10-linear-groupstep-schedule-plan.md)
- Log: [`2026-09-10-linear-groupstep-schedule.md`](../../logs/execution/2026-09-10-linear-groupstep-schedule.md)

## Mechanism

L-EM1 built a **exact** local metric for the mantissa-code descent — `G = h_inv^{-1} − cI`
rebuilt from the parent state, cross matrix `H = (Ŵ − W)ᵀŴ`, ideal target
`J(X) = ‖(X − T)Ŵᵀ‖²`, `T = X_ref C G^{-1}` — and then walked the 640 (in=2560) or 1024
(in=4096) natural 4-element groups **one at a time**, re-verifying each whole per-row move
against the exact full-row quadratic and keeping it only when strictly negative. That
precision was established: `+0.107791` on shard0, `−21.80%` true output error on a single
case. What closed L-EM1 was **time**: `+0.5671 s/call` sequential overhead, projecting
`≈382 s` official against a 300 s gate. Micro-optimisation could not rescue it — the cost
is dispatch-bound at ~25 µs per small-tensor launch, and even an analytic closed form still
projected `+44.7 s` against a 19 s budget.

L-EM2 keeps **every** fixed choice of L-EM1 — the metric, the cross matrix, the target, the
pm1 candidate set, the exact row-joint acceptance rule — and changes only the **descent
schedule**: step `k` proposes group `k` of *every* 64-block simultaneously, so a pass is
**16 sequential iterations** instead of 640/1024. Groups are visited in natural order.
Monotonicity is preserved by construction exactly as before, because each step's whole
per-row move is still re-verified against the exact full-row quadratic and a rejected row
simply does not move.

The schedule was chosen from a measured frontier (`probe_frontier.py`, real 4B data, read-only
CPU) that also killed the obvious alternatives: proposing a whole layer at once (`jacobi`)
is rejected on **every** row under every pass count, so the value of the mechanism is the
sequential refresh, not the proposal; moving a whole 64-block per step (`blockseq`) recovers
almost nothing. `groupstep` recovers **88%** (in=2560) / **64%** (in=4096) of the fully
sequential first pass with 1/40 the iterations.

## Calibration hook: the Cholesky inverse is gone

`cholesky_inverse(cholesky(h_inv))` was the single largest term in the calibration hook, and
it was pure waste there: its only purpose was to read off the scalar
`c = mean(diag(h_inv^{-1} − gram))`. Splitting that into two means removes it entirely:

- calibration stores `mean(diag(gram))` — a scalar for a scalar, so **stored size is unchanged**;
- dynamic time computes `c = mean(diag(h_inv^{-1})) − mean(diag(gram))`, and `h_inv^{-1}` is
  built there anyway.

Identical in exact arithmetic, differing only by ~1e-7 of fp32 rounding in the two means.
`verify.py` control C measures `diag_gap = 0.000e+00` and `path_rel = 0.000e+00` on both a
synthetic and the real state, i.e. the split ridge reproduces the fused one **bit for bit**.
Every faster-inverse alternative was measured and rejected (`linalg.inv` 40.87 ms,
`solve` 42.13 ms, `lu_factor+lu_solve` 43.72 ms, `inv_ex` 40.27 ms, `cholesky_solve`+`cholesky`
≈32.6 ms, all against `cholesky_inverse`'s 39.9 ms at in=4096; the ranking is stable across
runs, the absolute values are not).

## Verification — `verify.py`, CPU, all controls PASS

- **A** the candidate is a byte-exact prefix extension of the parent; the six APIs import
  standalone outside the repo; the Attention four APIs are character-identical to the parent.
- **B** parent-off control: with `state["em1"]` removed the dynamic output is **bit-identical**
  to the parent. `in_features=4160` takes the out-of-scope bypass and stores no metric. The
  parent's five fields are untouched by calibration.
- **C** metric pipeline: `H` bit-identical to an independent recomputation, `gram_diag_mean`
  gap `0`, `G` recovery `1.25e-05` synthetic / `1.16e-06` real, and the **live** `_em1_metric`
  code path agrees with an independent recomputation bit-for-bit (`path_rel=0`, `path_h_rel=0`).
- **D** monotonicity and the cost identity on the real case `layer0/q`, rows 128:
  `L_before=1.932880e+03 → L_after=1.563597e+03`, `dL = −19.1053%`, against the probe
  frontier's `ideal/groupstep/pm1/p1 = −19.0998%` (gap `5.5e-5`); self-reported
  `predicted_cost=−3.692835e+02` against true `dL=−3.692832e+02` (rel `8.7e-7`); analytic
  gradient `+2.076143e+00` against a finite difference `+2.076145e+00` (rel `1.3e-6`).
- **E** determinism (two calls bit-identical) and `torch.save/load` state round-trip.
- **F** schedule shape: `accepted_steps=16 == K*16`. `expected_passes` is passed in by the
  caller rather than read off the module, so a drifted `K` cannot silently agree with itself.

## Local result — six shards, 336 cases, paired

`artifacts/proxy_v3/linear-em2-sixshard/candidate/manifest.json`

| shard | delta gain mean | median | +/−/0 |
|---:|---:|---:|---|
| 0 | +0.088372 | +0.080715 | 48/0/8 |
| 1 | +0.086232 | +0.089640 | 48/0/8 |
| 2 | +0.071936 | +0.065280 | 48/0/8 |
| 3 | +0.077731 | +0.077613 | 48/0/8 |
| 4 | +0.072229 | +0.065865 | 48/0/8 |
| 5 | +0.080221 | +0.069129 | 48/0/8 |
| **equal-shard mean** | **+0.079454** | | **288/0/48** |

Candidate overall mean `0.608720` vs baseline `0.529266`. **288 of 336 paired cases improve,
none regress, 48 are unchanged** — and those 48 are exactly the out-of-scope `proj` role
(6 shards × 8), bit-identical to the parent, which is the evidence that the entire gain comes
from the mechanism and not from the wrapper. No shard is negative.

On the same shard0 panel L-EM1's fully sequential descent scored `+0.107791`, so **L-EM2 at
K=1 captures 82% of L-EM1's gain with 1/40 the sequential iterations**, matching the probe's
88% (in=2560) prediction in direction.

### On the local `Decision: reject`

The local analyzer reports `reject` on all six shards. Its gate is
`delta_mean > 0 and L1 < 0.02` — an eligibility screen aimed at **small** local trends — and
the tool's own policy states that `local eligibility is not promotion`. L-EM1's shard0
tripped the same screen at `L1 = 0.107791`. A mechanism that deliberately makes a large change
exceeds that `L1` by design. Per the standing rule, the local Δ is a diagnostic; this archive
does not claim a local pass.

## Time — K = 1, projected ≈ 294 s

The plan card pre-registered K = 2 with a **time-driven fallback**: if the real paired
measurement projects above 296 s official, drop to K = 1; if K = 1 still exceeds it, close the
card as time-infeasible.

**Dynamic API** (`time_paired.py`, same process, both arms from one calibration, arms
alternated per repeat with `cuda.synchronize` inside the timed region, 8 repeats, 9 cases):

| in_features | K=1 s/call | K=2 s/call |
|---|---:|---:|
| 2560 | +0.0385 | +0.0584 |
| 4096 | +0.0844 | +0.1238 |
| **projected (120×2560 + 24×4096)** | **+6.6 s** | +10.0 s |

The minimum-based estimator (least contended) gives +6.3 s / +9.1 s — both estimators bracket
the same answer, which is the evidence that the delta is real rather than machine load. The
out-of-scope `proj` control (in=9216, `changed_mantissa=0`) measures the noise floor at
0.003–0.02 s.

**Calibration hook** (`profile_compile_hook.py`, isolated: the hook called directly on the real
shard state with the parent excluded, 12 repeats): 41.31 ms at in=2560, 52.04 ms at in=4096,
projecting **+5.8 to +6.2 s** over the 144 in-scope calibrations.

| K | root | dynamic | calibration | total | margin |
|---|---:|---:|---:|---:|---:|
| 2 | 281 s | +10.0 s | +6.0 s | **≈297.5 s** | ~2.5 s |
| **1 (selected)** | 281 s | +6.5 s | +6.0 s | **≈294 s** | ~6 s |

K = 2 exceeded the pre-registered 296 s line under **both** measurement runs (298.3 s and
297.5 s), so the fallback fired. That is not only rule-following: a 2.5 s margin against a
measurement noise floor of comparable size is not a margin, and an overrun past 300 s scores
zero. K = 1 gives up 7 points of relative loss reduction (−19.10% vs −26.06% at in=2560) to
buy ~3.5 s.

Two independent checks support the dynamic figure. First, the paired prediction is
reproduced across processes: over all 336 six-shard cases the cross-process scoring delta is
**+0.0376 s/call** against a paired prediction of **+0.0400** — 6% agreement on 336 cases.
(Shard0 alone showed +0.064 s/call, 1.6× high, traced to thermal drift: the local baseline
scores off a calibration **cache hit** while the candidate scores right after 140 s of
calibration. That confound does not carry to the official comparison, where both the recorded
281 s root run and the candidate run include a full calibration.) Second, the cost is
**dispatch-bound** (CPU-side small-tensor launches), which is why it is nearly
rows-independent — 0.0385 s/call at rows=128 and 0.0483 s/call at rows=512 — and does not
float with GPU load.

### The conversion convention matters: 286.5 s vs 294 s

The 294 s above uses the plan card's **naive additive** convention — local API seconds added
1:1 onto the official root. But this project ran a **decomposition regression** on the same
question (`docs/official-local-fitting-analysis-2026-09-04.md` §4; 21 versions, `R² = 0.799`,
`MAE = 10.1 s`):

```
T_official ≈ 170.3 + 0.1154·W_calib + 0.6939·A_calib + 0.7344·dyn_act − 1.5837·dyn_qkv
```

`dyn_act` and `W_calib` are exactly the two APIs this card changes. Applied differentially:

| term | local increment | official coefficient | official increment |
|---|---:|---:|---:|
| dynamic activation | +6.5 s | 0.7344 | **+4.8 s** |
| weight-calibration hook | +6.0 s | 0.1154 | **+0.7 s** |
| total | +12.5 s | | **+5.5 s** |

The official machine is Kunpeng 920B: cheap for large matrices, expensive for small tensors,
so a local second of weight calibration is worth only 0.115 s officially. That gives
**≈286.5 s**, margin ~13.5 s.

The regression is known to underestimate newer versions by ~20 s (v182 residual +21 s,
v183 +19 s) and its negative `dyn_qkv` coefficient is a collinearity artifact, not a speedup.
The differential use is more robust than the absolute one — intercept and residual cancel —
but the coefficients still carry fit uncertainty. So the honest statement is a **bracket:
286.5–294 s**, both inside the gate, margins differing by a factor of two. Neither is claimed
as comfortable.

> **Recorded, not acted on:** under the fitted conversion, K = 2 would project to
> `281 + 7.3 + 0.7 ≈ 289 s` and would **not** have tripped the 296 s fallback line; under the
> plan card's convention it projected 297.5 s and did. K = 1 is therefore a conclusion of the
> card's convention, and this archive records it that way rather than presenting it as forced
> by the physics. K was **not** re-selected after the measurement: the pre-registered rule
> fired under the convention it was written in, and re-picking K once the numbers are in is
> precisely what pre-registration exists to prevent. K = 2 is a lever for a later card.

**Time risk is stated, not hidden.** Official time is the only gate, and this project's own
record says it is not reliably predictable from local marginals. The closest anchor is v222,
measured officially at **293 s** for a **+12 s** marginal — essentially this candidate's
+12.5 s — but v229 timed out on a mechanism whose marginal cost should have been near zero.
The two contradict each other. The increments here are measured directly on two of the six
timed APIs rather than extrapolated, but neither conversion makes the 300 s gate comfortable.

## Open lever (not taken)

The remaining lever is to store `gram` as the metric instead of rebuilding it from `h_inv`,
which would remove the inverse from the dynamic side too. It costs +n² host RAM — ≈**+4.76 GB**
across the 144 in-scope states, on top of the ≈4.76 GB of `H` already held — and the local
shard0 process already peaked at a 20 GB working set. Not worth risking an OOM for ~5 s.
Recorded as the L-EM3 candidate.

## Official result

None. `official_status: PENDING`; score and time are `null`. No official submission has been
made for v230 — the user runs batched official evaluations.
