# HiF4 Quantization Competition Workspace

> Data snapshot: 2026-08-31. Current facts are keyed by this file, the latest
> evaluation log, and the `solution.py` SHA.

Development workspace for the Huawei 2026 NVFP4-to-HiF4 algorithm track.
The root `solution.py` is the only active submission file. Historical
candidates live under `solutions/` and are never imported at runtime.

Chinese version: [README.md](README.md)

## Current status

- **Official evaluation (third revision, 2026-08-31): the organisers changed the
  scoring weights and reduced the weight of Linear cases**, so official totals
  have dropped substantially. `A@W` fitting remains unrestricted; the only hard
  constraint is the end-to-end runtime **below 300 seconds**. The current
  confirmed official anchor is **v84: `16517 / 252.563s` (official pass,
  < 300s)**. New-weight totals must not be compared directly with old-weight
  scores (the 20k+ v66/v72/v74 numbers). See
  [`v84 official result`](logs/execution/2026-08-31-v84-official-result.md).
- Old-weight official records (historical reference only, not comparable to the
  new weight): v074 / C75 `22750 / 239.387s`, v072 `22662 / 226s`,
  v066 / C66 `22557 / 217.2s`, v051 / C47b `22451 / 234s`, and v031 / C39-FW plus
  v034 / C41b both at `21864` (`161.3s` / `159.4s`).
- The official panel remains **250 Linear cases + 200 Attention cases**; official
  absolute scores vary with the weight scheme, and local runs never duplicate
  cases or fit official absolute scores.
- External reference: the public [`youxilee/hif4`](https://github.com/youxilee/hif4)
  repository reports `24153 / 239s` under the old user-confirmed protocol. Its
  code is not imported; an exact v2.7 CPU diagnostic run has a highest single-model
  Qwen native total of `369.527269` and a like-for-like Qwen panel of `250.327102`
  (the local benchmark). The five-model sum `1085.743597` is diagnostic only and
  cannot be converted to the official score.
- Historical v024 scored `16043 / 173.8s`, but its Linear output-supervision
  path used output information for activation-side selection. Under the rules
  in force then, that `A@W -> Q(A)` use was non-compliant, so it was not used as
  a parent. **The official evaluation (revised 2026-08-31) removed every `A@W`
  fitting restriction and now constrains only end-to-end runtime**; old
  compliance rulings no longer apply as current constraints.
- The root `solution.py` is **v127: the v106 Linear path + the variable-length PAWV
  calibration fix**, kept as a research candidate against the official Attention shape
  risk. On 2026-08-31 the L3–L6/C1 experimental mechanisms (Global Activation-LRH,
  final-Gram gates, GALS, block-local permutation, rank-16/wide rank-4 factors,
  `G_64` hierarchy, structured factors, C1a–C1c) were **removed from the root file**;
  they survive only in `solutions/` archives and historical logs. The new
  `sampled-means-v1` profile (Qwen, 8 layers, 7 roles, 4 windows, 224 Linear + 32
  Attention cases) measures Linear mean `0.509408`, Attention mean `0.828395`, local
  API `151.136s`, wall `161.840s`; under the same sampling plan v74 measures
  `0.440305 / 0.671106 / 218.619s / 229.485s`. These four values are the current
  local A/B primary results. The v127 full-layer legacy run (`453.102s`) is retained
  for history only; never map a local 300-second reading onto the official limit.
  Itemized results, archive audits, and reproduction configs live in the [current status
  report](docs/current-solution-status.md), the [algorithm inventory](docs/algorithm-inventory-and-directions.md),
  the [archive implementation audit](docs/archive-implementation-audit.md),
  and [`solutions/README.md`](solutions/README.md). Next steps follow the
  [single active optimization plan](docs/superpowers/plans/2026-08-31-hif4-active-c1-structured-linear-plan.md):
  C2 low-cost cross-model guardrails, then C3 state/time compression; the trimmed
  mechanisms may only be re-planted afterwards within a viable budget.
- Current source SHA256:
  `75F21B7BE3630FFEFEAF2883BB699CE4901DF1BF6C0B39DD6E40F253561E32C0` (normalized LF;
  identical to the `solutions/20260831_v127_v106-pawv-variable-length-safe_scoreNA_timeNA/`
  archive snapshot).
- The active local evaluator is `real_model_suite.py` with the default
  `sampled-means-v1` profile on Qwen2.5-0.5B: it reports only the sampled
  Linear/Attention mean gains, writes the full layer/role/window/seed plan into
  `sample_plan`, and treats other models as explicit, independent guardrails that are
  never summed. Legacy `official_flow_total`/`panel_score` fields remain in JSON for
  compatibility but are no longer the primary metrics. See the
  [local metric calibration](logs/execution/2026-08-31-local-metric-calibration.md)
  for the unified protocol and official-anchor fitting.

Local time and scores are for paired candidate comparison only and are never
reported as official results. Every official result must be archived together
with the exact submitted SHA, score, and runtime.

## Data and plan governance (required)

### Keep data current

The “Current status” section above, [`solutions/README.md`](solutions/README.md),
[`docs/current-solution-status.md`](docs/current-solution-status.md), and the latest
execution log form the current fact snapshot. Every local evaluation, official result,
or active `solution.py` change must update in the same commit:

1. the data date, model/data revision, full command, and cache mode;
2. Linear/Attention components, panel, API time, and source SHA256;
3. the `solutions/README.md` ledger, current status report, and execution log;
4. for an official result, the submitted SHA, score, runtime, and date; otherwise keep `NA` and never substitute a local value.

When documents disagree, use this order: root `solution.py` plus the latest reproducible
JSON/log → `solutions/README.md` → current status report → other research documents.
Archived plans and old logs are historical only. Refresh each document’s update/data-snapshot
date whenever its numbers change; do not leave an old snapshot label in place.

### Writing and executing plans

There may be exactly one active optimization plan in
[`docs/superpowers/plans/`](docs/superpowers/plans/). The current file is
[`2026-08-31-hif4-active-c1-structured-linear-plan.md`](docs/superpowers/plans/2026-08-31-hif4-active-c1-structured-linear-plan.md).
When executing an optimization, consult **only this active plan**, the current root,
latest evaluation data, and the official rules. Files under
`docs/superpowers/archive/plans/` are read-only history and are never next-step instructions.

Plan lifecycle rules:

1. Before creating a plan, verify that `plans/` has no second `.md` besides `README.md`. To change the main line, move the old plan to `archive/plans/`, create the new active plan, and update both READMEs in the same commit.
2. Every step must state its hypothesis, code entry point, model/data, acceptance metric, expected artifacts, and failure handling. After execution, immediately record the actual result, source SHA, log link, and `done/rejected/blocked` status.
3. Archive every experiment—success, failure, timeout, or unsubmitted—using the candidate workflow. A result without complete source, SHA, or configuration is `non-reproducible`.
4. Archive a plan immediately when it is complete, superseded, explicitly stopped, or repeatedly blocked. Never continue adding new “next steps” to an archived file or leave multiple “current/active” claims.
5. Do not rewrite archived conclusions. For an implementation bug or data error, add an audit note or a new active plan and record the affected scope in the index.

Before and after execution, at minimum check:

```powershell
Get-ChildItem docs\superpowers\plans -File -Filter *.md |
  Where-Object Name -ne README.md
git diff --check
```

The first command must return exactly one active plan. If it returns zero or more than one,
clean up the plan directory before running an algorithm experiment.

## Does the local evaluator track the official direction?

The following is the legacy Qwen panel compatibility table, kept for historical
anchor tracing only; current ranking always uses the `sampled-means-v1`
Linear/Attention means and the corresponding sample plan:

| Candidate | Official score | Qwen panel (local relative score) |
| --- | ---: | ---: |
| C39 | 21864 | 230.096230 |
| C41b | 21864 | 230.096230 |
| C47b | 22451 | 237.541351 |
| C66 | 22557 | 238.282409 |
| v72 / C74 | 22662 | 240.683147 |
| v74 / C75 | 22750 | 242.505358 |

These panel values only demonstrate historical local ordering direction, not a
conversion to official absolute scores; the official-anchor fitting diagnostics and
their applicability bounds are in the
[local metric calibration](logs/execution/2026-08-31-local-metric-calibration.md).
The external `youxilee/hif4` Qwen panel reference is `250.327102`; the legacy v125
precision parent measured `295.847849` on the old full-layer panel (accuracy upper
bound evidence only — those mechanisms are no longer in the root file, and all these
scores predate the 2026-08-31 official weight change).

## Revised official anchors (2026-08-29; weights changed 2026-08-31)

| Submission | Score | Runtime | Status |
| --- | ---: | ---: | --- |
| v031 / C39-FW | 21864 | 161.3s | compliant archive (old weight) |
| v034 / C41b | 21864 | 159.4s | compliant archive (old weight) |
| v051 / C47b | 22451 | 234s | previous local official result (old weight) |
| v066 / C66 | 22557 | 217.2s | previous official control (old weight) |
| v072 / C74 | 22662 | 226s | previous official result; Attention passed (old weight) |
| v074 / C75 | 22750 | 239.387s | old-weight official baseline; Attention passed |
| **v84 / C84** | **16517** | **252.563s** | **official pass under the new scoring weights (Linear weight reduced); < 300s** |
| `youxilee/hif4` | 24153 | 239s | external old-weight official reference; local highest Qwen native `369.527269`, panel `250.327102`; five-model `1085.743597` is diagnostic only |

> **Weight change (2026-08-31)**：the organisers reduced the weight of Linear
> cases, so the official total dropped sharply (v84 `16517` vs old-weight v74
> `22750`). The two weight schemes cannot be converted into each other; the
> exact coefficients were not disclosed, and local runs never fit official
> absolute scores. See [`v84 official result`](logs/execution/2026-08-31-v84-official-result.md).

The official runtime limit has been revised to **5 minutes (300 seconds)**
(2026-08-31); the passing versions in the table were evaluated under the
former 420s ceiling, and their times remain below the new 300s limit. Legacy
values such as `14613 / 159.2s` remain historical records from the old panel.

## Hard competition constraints

1. **The official evaluation (revised 2026-08-31) imposes no `A@W` fitting
   restrictions of any kind.** Both offline
   `hif4_calibration_and_quantize_weight` and online activation quantization
   may freely use `A@W`, outputs, or residuals to optimize `Q(W)` / `Q(A)`;
   there is no information-source restriction. The only hard constraint is the
   end-to-end runtime.
2. **The scoring weights were changed on 2026-08-31 to reduce the weight of
   Linear cases**, so official totals dropped substantially; new-weight totals
   are not comparable with old-weight scores. Local ranking still uses the
   `sampled-means-v1` Linear/Attention means and is unaffected by the official
   total-weight change.
3. Produce legal HiF4 fields and keep the API, state, shape, dtype, and device
   behavior valid.
4. Keep the final official evaluation strictly below `300s` (5 minutes).
5. Do not tune from holdout or official-score feedback.

There are no fixed gain, coverage, beam, per-component non-regression, or
intermediate runtime gates beyond these rules. Diagnostic development runs may
exceed 300 seconds. Once an accuracy signal is found, optimize the algorithm
and implementation to fit the final runtime limit.

For the current accuracy-first phase, runtime is recorded but is not an
acceptance gate. The `<300s` requirement becomes a hard gate again only when a
validated mechanism enters submission compression.

## Current algorithm

The root is **v127: the v106 Linear chain + Attention B1/B2 + the variable-length
PAWV fix**. The L3–L6/C1 experimental mechanisms were trimmed from the root file on
2026-08-31 and survive only in archives; v086/C86 remains an immutable historical
archive.

### Linear

1. Reconstruct the floating-point reference from NVFP4 scales and E2M1 data.
2. BOAT searches RMS diagonal balancing and 4/8/16/64-dimensional signed
   Hadamard blocks using operand-local errors only; it never constructs a
   Linear output.
3. Cross-fold Weight-HSDQ uses `A.T @ A` block Hessians, 15 signed levels,
   top-two 64-dimensional blocks and one coordinate sweep. A candidate made on
   one calibration fold must improve the other before admission.
4. Online Activation-HSDQ uses static transformed-weight Gram blocks to choose
   hierarchy/offset and refine at most 128 blocks for two sweeps. Only static
   Gram and legal BOAT configuration are stored in `activation_state`.
5. The expansive-FFN CAT balance (`rows > channels`, fixed `α=0.25`) improves
   `fc_gate` without adding state fields.

### Attention

The active path searches reciprocal RMS balancing, K-centering, GQA alignment,
and shared 16/32/64-dimensional signed Hadamard transforms. A cheap proxy scans
all candidates, then the strongest four are re-evaluated through the deployed
quantizer and real non-causal Attention output. Q/K state stores only static CPU
Gram, importance, integer block/seed, and signs; V remains an independent legal
HiF4 encoding. The old C86 experiment flags, Segment-CVaR, and non-beneficial
V-importance branches are not present in the root file.

## Development principles

- Paired scores from the deployed path decide candidates. Oracles and local
  losses are diagnostic tools for ranking and explanation.
- Do not veto implementation with arbitrary percentage thresholds unless a
  strict mathematical impossibility has been proved.
- It is valid to remove redundant work and reallocate the saved compute in one
  coherent mechanism; perform ablations afterward.
- Preserve the full accuracy-runtime Pareto curve. A negative run does not
  prove the whole competition space unreachable.
- Small stable gains may accumulate; no candidate must clear a fixed gain.
- Archive failures, but scope each negative conclusion to the implementation
  and configuration actually tested.

## Repository layout

```text
solution.py                         only active submission file
evaluator/real_model_suite.py       Qwen-first real-data evaluation, fixed-panel ranking, and cache
evaluator/reference_hif4.py         independent official scoring protocol, baseline, and validation
evaluator/nvfp4_sim.py              NVFP4 encoding simulation
evaluator/real_data_eval.py         shared loading/timing/scoring tools and the legacy single-model entry point
evaluator/synthetic_attention_eval.py
                                    576-case Attention safety matrix (property diagnostics, not ranking)
evaluator/linear_compliance_guard.py
                                    static/runtime Linear compliance checks
evaluator/linear_error_decomposition.py
                                    Linear error attribution diagnostics
tests/                              release, format, compliance, algorithm tests
solutions/                          immutable candidate archives
artifacts/real_model_suite/         evaluation JSON results; cache/ holds local model snapshots, not committed
logs/evaluations/                   evaluation run reports (path must be given explicitly)
logs/candidates/                    candidate official results and diagnosis reports
logs/execution/                     execution logs and calibration records
docs/current-solution-status.md    current root algorithm, measurements, and score attribution
docs/real-model-evaluator.md        evaluator usage guide
docs/research/                      literature survey
docs/superpowers/plans/             the single active implementation plan
docs/superpowers/specs/             designs and specifications
docs/superpowers/archive/plans/     superseded plans, historical only
```

## Running evaluations

Create the project environment:

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install -r evaluator\requirements.txt
```

### Recommended: Qwen primary panel (cached)

Use this command for daily candidate comparisons. It evaluates Qwen2.5-0.5B and
projects the means onto the fixed 250 Linear + 200 Attention panel. The `read`
mode requires a matching snapshot and never downloads a model implicitly:

```powershell
.\.venv\Scripts\python -u evaluator\real_model_suite.py `
  --models qwen2.5-0.5b --candidates c39 c41b c47b c66 `
  --solution solution.py --candidate-name active `
  --panel-profile qwen-official --primary-model qwen2.5-0.5b `
  --device cpu --algorithm-device cpu --cache-mode read `
  --seq 128 --calib 2 --test 4 `
  --output artifacts\real_model_suite\qwen-panel-YYYYMMDD.json `
  --report logs\evaluations\qwen-panel-YYYYMMDD.md
```

If CUDA is available, change both device flags to `cuda`. If the cache is
missing, run the capture step below first.

The default daily ranking command uses the fast, reproducible `sampled-means-v1`
profile:

```powershell
.\.venv\Scripts\python -u evaluator\real_model_suite.py `
  --models qwen2.5-0.5b --evaluation-profile sampled-means-v1 `
  --sample-layers 8 --sample-test-windows 4 --sample-seed 20260831 `
  --device cpu --algorithm-device cpu --cache-mode read `
  --solution solution.py --candidate-name active `
  --output artifacts\real_model_suite\active-sampled.json `
  --report logs\execution\active-sampled.md
```

This pins 224 Linear and 32 Attention cases and reports only
`mean_scores.linear_mean` and `mean_scores.attention_mean`. Changing any
profile, seed, layer/window count, device, cache, or data revision produces a new
group that cannot be compared directly with previous results. Key result fields:

| Field | Purpose |
| --- | --- |
| `results[*].mean_scores.linear_mean` | the only Linear primary metric (0–1 gain mean) |
| `results[*].mean_scores.attention_mean` | the only Attention primary metric (0–1 gain mean) |
| `results[*].sample_plan` | actual layer/window/role/seed; must match exactly before comparing |
| `results[*].timing.local_api_total_seconds` | local six-API cumulative time |
| `results[*].timing.wall_seconds` | local wall time including scheduling/reporting |
| `results[*].official_flow_score` / `panel_score` | legacy compatibility fields, not primary metrics |

Commands with `--solution` exit with code `2` only when local results are
incomplete, illegal, or non-finite; they still write JSON and Markdown. A local
API time above 300s is never reported as an official timeout. Old full-layer
numbers are legacy; use the sampled profile for daily ranking, and confirm
official time only through the official platform.

Single-model quick evaluation (gpt2-small, cache-first):

```powershell
.\.venv\Scripts\python -u evaluator\real_model_suite.py `
  --models gpt2-small --solution solution.py --candidate-name active `
  --panel-profile qwen-official --primary-model gpt2-small `
  --device cpu --algorithm-device cpu --cache-mode auto `
  --seq 128 --calib 2 --test 4 `
  --output artifacts\real_model_suite\quick-YYYYMMDD.json `
  --report logs\evaluations\quick-YYYYMMDD.md
```

GQA example (Qwen2.5-0.5B ships 14Q/2KV heads with RoPE):

```powershell
.\.venv\Scripts\python -u evaluator\real_model_suite.py `
  --models qwen2.5-0.5b --solution solution.py --candidate-name active `
  --panel-profile qwen-official --primary-model qwen2.5-0.5b `
  --device cpu --algorithm-device cpu --cache-mode auto `
  --seq 128 --calib 2 --test 4 `
  --output artifacts\real_model_suite\quick-qwen-YYYYMMDD.json `
  --report logs\evaluations\quick-qwen-YYYYMMDD.md
```

Synthetic Attention matrix:

```powershell
.\.venv\Scripts\python evaluator\synthetic_attention_eval.py `
  --solution solution.py
```

Clean-root release regression (format, compliance, reference codec, and the
real-model evaluator):

```powershell
.\.venv\Scripts\python.exe -m pytest -q `
  tests/test_reference_hif4.py tests/test_linear_compliance_guard.py `
  tests/test_real_model_suite.py --basetemp=.tmp_pytest\clean-root
```

The current environment reports **36 passed** (re-measured 2026-08-31 after the
sampled-means tests were added). `test_jdrq.py`,
`test_weight_cross64.py`, `test_weight_full64.py`, and `test_release_candidate.py`
still contain historical assertions for removed C86/JDRQ helpers, experiment
flags, or state schemas. They are not clean-root release gates; do not use the
failure count from an unfiltered `pytest -q` run to judge the active algorithm.
If one of those directions is restarted, rewrite its tests against the current
API/state before adding it to the release command.

To run only the real-model evaluator (with an ignored repository-local temp
directory):

```powershell
.\.venv\Scripts\python.exe -m pytest -q tests/test_real_model_suite.py --basetemp=.tmp_pytest\readme-verify
```

### Candidate testing order and result capture

Modify only the root `solution.py` for each experiment. Run syntax, compliance,
and single-model real-path checks before multi-model comparison. Do not edit
historical sources under `solutions/`.

1. **Fast pre-commit checks**

   ```powershell
   git diff --check
   .\.venv\Scripts\python -m py_compile solution.py evaluator\real_model_suite.py evaluator\reference_hif4.py evaluator\linear_compliance_guard.py
   .\.venv\Scripts\python.exe -m pytest -q `
     tests/test_reference_hif4.py tests/test_linear_compliance_guard.py `
     tests/test_real_model_suite.py --basetemp=.tmp_pytest\clean-root
   ```

2. **Test the active root `solution.py`**

   This is a fast `gpt2-small` smoke test for output format, Linear, Attention,
   and local runtime. It is not the official-direction primary ranking; use the
   Qwen command above for candidate promotion:

   ```powershell
   .\.venv\Scripts\python -u evaluator\real_model_suite.py `
     --models gpt2-small --candidates c39 `
     --solution solution.py --candidate-name active `
     --panel-profile qwen-official --primary-model gpt2-small `
     --device cpu --algorithm-device cpu --cache-mode read `
     --seq 128 --calib 2 --test 4 `
     --output artifacts\real_model_suite\active-YYYYMMDD.json `
     --report logs\evaluations\active-YYYYMMDD.md
   ```

   The default evaluation is the Qwen `sampled-means-v1` profile; compare only
   `mean_scores.linear_mean` and `mean_scores.attention_mean`. Every result must
   record the sample seed, layer/window indices, source case counts, Local API,
   Wall, device, and source SHA256. Legacy `panel_score`/`official_flow_total`
   fields exist only for reading historical JSON. The smoke command above
   explicitly uses gpt2-small; local scores are only for paired A/B ranking and
   must not be entered as Official Score. Pair promotion runs with
   `--candidates c39 c41b c47b c66`, and record the full command,
   source/target case counts, total API time, and source SHA256.

3. **Capture real multi-model forward data once**

   `real_model_suite.py` covers GPT-2 small/medium, OPT-125M, Pythia-160M,
   and Qwen2.5-0.5B by default, and compares the revised C39/C41b/C47b/C66
   anchors. Qwen drives the primary score; other models are soft guardrails.
   Capture model data first so each candidate does not repeat model forward
   execution:

   ```powershell
   .\.venv\Scripts\python -u evaluator\real_model_suite.py `
     --device cpu --algorithm-device cpu --cache-mode write --capture-only `
     --seq 128 --calib 2 --test 4 `
     --output artifacts\real_model_suite\cache-capture-YYYYMMDD.json `
     --report logs\evaluations\cache-capture-YYYYMMDD.md
   ```

   Replace `YYYYMMDD` with the actual run date. On a CUDA host, change both
   device flags to `cuda` to shorten capture. Snapshots are stored in
   `artifacts/real_model_suite/cache/` and are not committed to Git. They
   contain real model weights, Linear inputs, real Q/K/V, token ids, model/data
   revisions, and window-validation metadata.

4. **Evaluate from cache only**

   After capture, candidate tests do not load a tokenizer/model, execute model
   forward, or access the network:

   ```powershell
   .\.venv\Scripts\python -u evaluator\real_model_suite.py `
     --candidates c39 c41b c47b c66 --solution solution.py --candidate-name active `
     --panel-profile qwen-official --primary-model qwen2.5-0.5b `
     --device cpu --algorithm-device cpu --cache-mode read `
     --seq 128 --calib 2 --test 4 `
     --output artifacts\real_model_suite\active-YYYYMMDD.json `
     --report logs\evaluations\active-YYYYMMDD.md
   ```

   In `read` mode, a missing cache, version/configuration mismatch, leaked
   window, or invalid tensor shape fails immediately; it never silently loads
   a model. `auto` is convenient for daily use: it reads a valid cache and
   recaptures a missing or stale one. `write` always refreshes; `off` neither
   reads nor writes. Changing sequence length, calibration/test counts, layer
   cap, model, or the pinned dataset revision requires a new cache.

5. **Compare local component means**

   Compare only the two means under the same sample plan; official-anchor
   ordering and score fitting are post-hoc diagnostics in a separate calibration
   report and never participate in candidate runs or absolute score conversion.
   The evaluator keeps legacy fields so historical JSON remains readable, but
   new reports no longer present them as primary scores.

   The current primary formula is:

   ```text
   score(case) = (MSE_STD - MSE_PLAYER) / MSE_STD
    linear_mean = mean((MSE_STD-MSE_PLAYER)/MSE_STD over sampled Linear cases)
    attention_mean = mean((MSE_STD-MSE_PLAYER)/MSE_STD over sampled Attention cases)
   ```

   Standard NVFP4/HiF4 dequantization, HiF4 parameter validation, and state
   validation are evaluator-owned. A candidate only needs the six official
   APIs. Evaluator-side `A@W` is formed only after the candidate has returned
   its quantized result and is never passed back as calibration data. Under the
   2026-08-31 revisions, the candidate may freely use `A@W` to optimize both
   `Q(W)` and `Q(A)`; the official side no longer restricts information sources.

   The task document does not include the source of the official "standard
   HiF4 quantizer." The current independent codec is the historically audited
   implementation and its SHA256 is recorded in every report. Replace it
   bit-for-bit and bump the protocol version when the official function is
   available.

6. **Record time but never fake official verdicts**

   `local_api_total_seconds` is the local six-API cumulative time and
   `wall_seconds` is the local wall clock; both support A/B only on identical
   hardware, cache, shapes, and sample plan. The official `300s` is the
   end-to-end limit over the official 450 cases on Kunpeng 920B hardware and
   can only be confirmed by the official platform.

### Candidate archiving steps

Archive every experiment—successful, failed, unsubmitted, or officially timed
out. Do not keep only improvements. Before archiving, freeze the root
`solution.py` bytes and the test evidence:

1. Allocate the next version number using
   `solutions/YYYYMMDD_vNNN_topic_scoreSCORE_timeTIME/`. If the official result
   is unknown, use `scoreNA_timeNA`; never put local score/runtime in the
   Official Score/Time fields or overwrite an original record later.
2. Copy the root source into the archive; the root file remains the only active
   submission file:

   ```powershell
   New-Item -ItemType Directory -Path solutions\YYYYMMDD_vNNN_topic_scoreNA_timeNA
   Copy-Item -LiteralPath solution.py `
     -Destination solutions\YYYYMMDD_vNNN_topic_scoreNA_timeNA\solution.py
   Get-FileHash -Algorithm SHA256 solution.py
   Get-FileHash -Algorithm SHA256 solutions\YYYYMMDD_vNNN_topic_scoreNA_timeNA\solution.py
   ```

   The two SHA256 values must match exactly. Never edit the archived source
   afterward.
3. Create `result.md` in the same directory. At minimum record the date,
   version/parent, the single algorithm change, hypothesis, complete test
   command and configuration, every Linear/Attention/runtime result, cache
   filename and dataset/model revisions, active source SHA256, official
   score/runtime, delta, status, conclusion, and next direction. When the
   cache is not committed, state that it must be recaptured according to this
   README.

   Use this minimum template and keep `NA` for unknown values:

   ```markdown
   # vNNN — topic

   - Date: YYYY-MM-DD
   - Parent: vNNN / commit
   - Change: one primary algorithm change
   - Hypothesis: why this change may improve accuracy
   - Test command: `full command`
   - Test config: model/data/cache/mode/layers/algorithm-device
   - Sample profile/seed/layer-window plan: ...
   - Local Linear mean / cases: ...
   - Local Attention mean / cases: ...
   - Local API seconds / Wall seconds / device: ...
   - Cache: filename, schema, dataset revision, model revision
   - Source SHA256: `...`
   - Official score: NA
   - Official runtime: NA
   - Status: `local-rejected` / `local-accepted` / `official-compliant-champion`
   - Conclusion: evidence-based decision
   - Next direction: next falsifiable experiment
   ```

4. Update the comparison table in `solutions/README.md` and the relevant
   execution log. When the official result arrives, append the submitted SHA,
   score, runtime, and date without replacing local evidence. The submitted
   file must have the same SHA256 as the archive.
5. Check the archive and tests, then commit:

   ```powershell
   git diff --check
   .\.venv\Scripts\python -m py_compile solutions\YYYYMMDD_vNNN_topic_scoreNA_timeNA\solution.py
   .\.venv\Scripts\python.exe -m pytest -q `
     tests/test_reference_hif4.py tests/test_linear_compliance_guard.py `
     tests/test_real_model_suite.py --basetemp=.tmp_pytest\archive-check
   git add solution.py solutions\YYYYMMDD_vNNN_topic_scoreNA_timeNA\solution.py `
     solutions\YYYYMMDD_vNNN_topic_scoreNA_timeNA\result.md solutions\README.md `
     logs\execution\YYYYMMDD-experiment.md
   git commit -m "archive vNNN candidate"
   git push origin master
   ```

   If the change only updates the evaluator or documentation, state clearly in
   the commit that active `solution.py` was unchanged.

## Records

- Current facts come from root `solution.py`, the latest execution log, and
  reproducible evaluator output.
- Historical versions and decisions are indexed in
  [solutions/README.md](solutions/README.md).
- The latest execution records are the
  [local metric calibration](logs/execution/2026-08-31-local-metric-calibration.md)
  and the [`v84 official result`](logs/execution/2026-08-31-v84-official-result.md).
- Candidate archiving follows
  [2026-08-26-solution-archive-workflow.md](docs/superpowers/archive/plans/2026-08-26-solution-archive-workflow.md).
- The archive implementation audit is in
  [docs/archive-implementation-audit.md](docs/archive-implementation-audit.md).
- Multi-model real data, cache modes, and compliance boundaries are documented
  in [real-model-evaluator.md](docs/real-model-evaluator.md).
- Official per-case summation, independent codec/validation, and ranking audit
  are documented in [real-model-evaluator.md](docs/real-model-evaluator.md).
- Superseded optimization plans were moved to
  `docs/superpowers/archive/plans/` and are not active instructions.
