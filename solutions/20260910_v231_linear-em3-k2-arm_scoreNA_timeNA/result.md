# v231 — Linear group-major exact-metric descent, K = 2 (L-EM3)

Status: **PENDING**. No official evaluation has been run. The L-EM3 card fixed this
outcome in advance — *"whether or not the projection clears 296 s, v231 is archived as
`PENDING`"* — because the card exists precisely to have the official machine price the
two pass counts, not to have the local measurement pick a winner.

- Parent: v202 Linear + v195 Attention complete root, as the card was written
- Parent SHA256: `56DC805D6E5A3AEF896DB8021045740292735725D688B48E3D4393E55EFCB2BD` (485072 B)
- Candidate SHA256: `ea79a1c12dc667142c620975aab188920fae7b29988c804f41c1a696cc5754f1` (505762 B)
- Version note: v231 is the **sibling of v230** — the same Linear mechanism at K=2 instead
  of K=1. The card registered this number as unoccupied; at archive time this is the only
  `v231` under `solutions/`. v230 was shared by both lines, so if the Attention line takes
  v231 too, quote the full directory name.
- Plan: [`2026-09-10-linear-k2-timed-arm-plan.md`](../../docs/superpowers/archive/plans/2026-09-10-linear-k2-timed-arm-plan-superseded.md)
- Log: [`2026-09-10-linear-em3-k2-arm.md`](../../logs/execution/2026-09-10-linear-em3-k2-arm.md)

## The root moved while this card was in flight

At 10:31 on 2026-09-10 the user committed `4104b9a`
*"feat: promote v230 Linear official 18428 in 292s"*. **v230 passed officially at
18428 / 292 s** and the root became the v230 candidate, `0F1AF6DB…`. This candidate was
built and paired before that swap.

The two files are **siblings**: both are pure appends of `56dc805d`, neither prefixes the
other, and they share the card parent's entire 485072 bytes. So this archive reports two
deltas, which answer two different questions:

| comparison | Δ (equal-shard mean) | cases | why it is recorded |
|---|---:|---|---|
| vs card parent `56dc805d` | **+0.106961** | 288 / 0 / 48 | same baseline as the L-EM2 archive's registered `+0.079454`, so the two are comparable |
| vs current root `0f1af6db` (= v230) | **+0.027507** | 286 / 0 / 50 | the K=2-over-K=1 increment — what a future official evaluation would actually measure |

`0.106961 − 0.079454 = 0.027507` closes bit-for-bit.

## Mechanism

L-EM2 established that L-EM1's exact metric can be made affordable by changing only the
**descent schedule** — step `k` proposes group `k` of *every* 64-block at once, so a pass
costs 16 sequential iterations instead of 640/1024 — while keeping every fixed choice: the
metric `G = h_inv^{-1} − cI`, the cross matrix `H = (Ŵ − W)ᵀŴ`, the ideal target
`J(X) = ‖(X − T)Ŵᵀ‖²` with `T = X_ref C G^{-1}`, the pm1 candidate set, and the exact
whole-row joint acceptance rule.

L-EM3 changes **one** thing: the pass count `K: 1 → 2`. Each pass re-derives the exact
gradient at its start rather than reusing the previous pass's. This is the L-EM2 card's own
pre-registered primary arm — that card took its time-driven fallback to K=1 because the
fallback rule fired under the naive additive convention, and L-EM3 submits the primary.
`K` was not re-selected after the fact, and `K ∈ {3,4,…}` was not scanned.

The mechanism is **monotone by construction**: every step's whole per-row move is
re-verified against the exact full-row quadratic and a rejected row simply does not move.
Raising `K` therefore carries time risk only, never accuracy risk. Control F certifies it
directly — `accepted_steps = 32 = K × 16`, against an `expected_passes` passed in by the
caller so a drifted `K` cannot silently agree with itself.

## Local accuracy

Six-shard proxy-v3, linear-only, 336 cases, two single-side `eval.py` processes paired out
of process. Candidate mean `0.63622687`, baseline mean `0.52926585`, difference
`0.10696102`.

| shard | n | Δmean vs card parent | Δmean vs current root | +/0/− |
|---|---:|---:|---:|---|
| 0 | 56 | +0.118274 | +0.029902 | 48/8/0 |
| 1 | 56 | +0.115701 | +0.029469 | 48/8/0 |
| 2 | 56 | +0.096028 | +0.024091 | 48/8/0 |
| 3 | 56 | +0.105419 | +0.027687 | 48/8/0 |
| 4 | 56 | +0.098018 | +0.025789 | 48/8/0 |
| 5 | 56 | +0.108326 | +0.028104 | 48/8/0 |

Every shard is positive and **nothing regresses**. The 48 zeros are exactly the
out-of-scope `proj` role (6 shards × 8 cases), bit-identical to the parent, so every gain
comes from the mechanism and none from the wrapper.

By role (shard 0, `proj` = 0): `o +0.2065`, `v +0.1736`, `k +0.1626`, `q +0.1465`,
`fc_up +0.0793`, `fc_gate +0.0594` — the same ordering as v230 at a larger amplitude,
which is what "the same mechanism, run one more time" should look like.

The local tool's `shard_decision` is `reject` for all six shards. That gate is
`delta_mean > 0 and L1 < 0.02`, an eligibility screen for *small* local trends, and the
tool's own policy states local eligibility is not promotion. A mechanism that deliberately
makes a large change exceeds that L1 by design; the archive does not claim a local pass.

### The split-process pairing is exact, not a fallback

The standard `eval.py` path holds both sides in one process and has `analyzer` pair them
directly. On this machine (31.8 GB RAM, one 8 GB GPU) that path was killed by the memory
watchdog **three times** — end of shard-0 scoring, mid-calibration, and 185 lines into
candidate scoring — so the two sides were evaluated in separate processes and paired by
`pair_sixshard.py` from the per-`case_id` gains.

That substitution was validated rather than assumed. For the v230 candidate, the 336
per-case gains are **bit-identical** between the in-process and single-side runs
(`max |difference| = 0.000e+00`), and the v230 delta recomputed from the same files is
`+0.079454`, bit-identical to the value the L-EM2 archive registered. Reusable finding:
in this project, split-process pairing is exact.

## Time

Measured in one process with the parent, K=2 and K=1 arms **alternated over the same
cases** under one shared calibration, with `cuda.synchronize` inside the timed region
(9 cases, median of 8):

- K=2 dynamic over the 144 in-scope official calls: **+9.2 s** (min-based estimator +8.9 s)
- K=1: +6.0 s; the **K=2 marginal over K=1 is +3.2 s**, paired on the same cases
- calibration hook, unchanged from L-EM2: **+5.8 to +6.2 s**
- noise floor, read off the out-of-scope `proj` control (`changed_mant = 0`): 0.011–0.013 s

### The v230 official reading settled the conversion convention

The card was written because two local-to-official conventions disagreed and neither could
be validated locally. Mid-card, the official reading for v230 (K=1, same mechanism) came
back at **292 s**, and it discriminates between them:

| convention | v230 prediction | error vs official 292 s |
|---|---:|---:|
| naive additive | 281 + 6.5 + 6.0 = **294 s** | **2 s** |
| fitted decomposition | 281 + 4.8 + 0.7 = **286.5 s** | 5.5 s |

The naive additive convention is **2.8× more accurate** on this reading, which makes it the
working convention for this mechanism family. Caveat recorded honestly: that is one data
point against a regression with MAE 10.1 s, and the mechanism differs from the ones the
regression was fitted on, so it is not a claim that the naive convention is globally better.

On that convention this candidate projects **292 + 3.2 = 295.2 s** — inside the 300 s gate
by about 5 s. The card, written against a 281 s root, had projected 296.2 s and treated the
0.2 s overshoot of its own 296 s line as unresolved; the root swap moved the picture in the
candidate's favour, and the K=1 arm's official number now anchors it.

## Verification

`workbench/full_solution/linear-em3-k2-arm/verify.py`, CPU: **`ALL L-EM2 CONTROLS PASSED`**.

| control | result |
|---|---|
| A — lineage, isolated import, Attention identity | candidate and live root agree on the card parent's full 485072 B; six APIs import standalone; Attention four APIs character-identical to the live root's source |
| B — parent-off bit identity, out-of-scope bypass | holds for the narrow layer and the `in=4160` bypass |
| C — `H` / `gram_diag_mean` / `G` recovery | `diag_gap = 0`, `path_rel = 0`, `path_h_rel = 0` on both synthetic and real layers |
| D — monotonicity, cost identity, gradient check | real `layer0/q` `dL = −26.0695%` vs the probe's `−26.0625%`; predicted cost matches the true `dL` to 3.1e-06; gradient matches a finite difference to 1.8e-06 (fp32 floor 3.5e-05) |
| E — determinism and state round-trip | both layers pass |
| F — schedule shape | `passes=2 accepted_steps=32 (cap 32)` |

### Two post-hoc repairs, disclosed

1. **The swap broke the controls' meaning, not their logic.** Control A read
   `ROOT/solution.py` as the parent and control B compared its off-path against it. After
   the promotion the live root is the *sibling arm*, which applies the metric rather than
   ignoring it, so control B failed with `parent-off control changed field sign`. Control A
   now asserts the candidate and the live root agree on the card parent's full 485072 bytes,
   and the parent module is recovered as a snapshot of the live root's first 485072 bytes.
   The §2 pre-registration table was not touched.
2. **The candidate carries a comment rewrite.** Beyond `_EM1_PASSES = 1 → 2`, the 5-line
   comment above that constant became 8 lines explaining why the K=2 arm is submitted.
   Comments do not execute, so §2's *mechanism* claim ("only `_EM1_PASSES` changes") holds
   literally, but the phrase "L-EM2 byte-for-byte unchanged" does not. Kept and disclosed
   rather than reverted because the measured `source_sha256` **is** `ea79a1c1` — reverting
   would make every result JSON here describe bytes other than the shipped ones, and the
   GPU re-run that would fix that buys nothing, since the change is non-executable.

## Decision

`LOCAL_POSITIVE_PENDING_OFFICIAL`. 288 of 336 paired cases improve, none regress, 48 are the
out-of-scope `proj` role; every shard is positive. The K=2 arm adds `+0.027507` over the K=1
arm already at root and costs about 3 s. Submitted for the user's batched official
evaluation; the card pre-registered `PENDING` either way.
