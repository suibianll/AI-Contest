# HiF4 Quantization Competition Workspace

Development workspace for the Huawei 2026 NVFP4-to-HiF4 algorithm track.
The root `solution.py` is the only active submission file. Historical
candidates live under `solutions/` and are never imported at runtime.

Chinese version: [README.md](README.md)

## Current status

- The official panel has been revised to **250 Linear cases + 200 Attention
  cases**. Because scores are summed per case, both scores and runtimes are
  higher than under the legacy panel and must not be compared directly.
- On the revised panel, the best official result among archived submissions is
  v066 / C66 at `22557 / 217.2s`; the previous v051 / C47b result was
  `22451 / 234s`. v031 / C39-FW and v034 / C41b both scored `21864`, at
  `161.3s` and `159.4s`, respectively.
- External reference: the public [`youxilee/hif4`](https://github.com/youxilee/hif4)
  repository reports `24153 / 239s` under the same user-confirmed protocol. Its
  code is not imported; an exact v2.7 CPU diagnostic run gives a five-model proxy
  total of `1085.743597` (Qwen2.5-0.5B: `369.527269`), which is not an absolute
  conversion to the official score. C66 remains `1596` points and `21.8s` away.
- Historical v024 scored `16043 / 173.8s`, but its Linear output-supervision
  path used output information for activation-side selection. That
  `A@W -> Q(A)` use remains non-compliant, so it is not a compliant parent for
  new work.
- The current root `solution.py` is v076/C77 all-shape gram64 plus C76.4
  GQA rotation. Qwen's local native total is `372.623675` and its panel score
  is `260.060290`; see [`solutions/README.md`](solutions/README.md) for the
  paired four-model measurements and mechanism details.
- Current source SHA256:
  `C87B61C8A4A9F869A43EFDEECF7734A0A810EA0E5621D51826EC5E56A31ED0E4`.
- The active local evaluator is Qwen-first: it projects frozen-corpus Linear
  and Attention means onto a fixed 250/200 panel, while other models remain
  soft guardrails. The raw `official_flow_total` is retained for compatibility
  diagnostics, but model layer counts no longer determine the primary ranking.

Local time and scores are for paired candidate comparison only and are never
reported as official results. Every official result must be archived together
with the exact submitted SHA, score, and runtime.

## Does the local evaluator track the official direction?

On the frozen five-model evidence, the Qwen primary panel reproduces the revised
official anchor ordering:

| Candidate | Official score | Qwen panel (local relative score) |
| --- | ---: | ---: |
| C39 | 21864 | 230.096230 |
| C41b | 21864 | 230.096230 |
| C47b | 22451 | 237.541351 |
| C66 | 22557 | 238.282409 |

Both orderings are `C39 = C41b < C47b < C66`; Qwen's panel Spearman is
`1.0000`, while the five-model raw sum is `0.9487`. This validates relative
direction only, not a linear conversion to official scores. The external
`youxilee/hif4` Qwen panel is `250.327102`, directionally above C66, but is not
a local anchor.

## Revised official anchors (2026-08-29)

| Submission | Score | Runtime | Status |
| --- | ---: | ---: | --- |
| v031 / C39-FW | 21864 | 161.3s | compliant archive |
| v034 / C41b | 21864 | 159.4s | compliant archive |
| v051 / C47b | 22451 | 234s | previous local official result |
| v066 / C66 | **22557** | **217.2s** | best local official result |
| `youxilee/hif4` | **24153** | **239s** | external official reference; local v2.7 CPU proxy `1085.743597` |

The revised official runtime limit is **7 minutes (420 seconds)**. Legacy values
such as `14613 / 159.2s` remain historical records from the old panel.

## Hard competition constraints

1. **Offline calibration may use `A@W` to optimize an offline quantizer,
   especially `Q(W)`.** It must not use `A@W`, its quantized output, or an
   output residual to fit, select, or infer the online activation quantizer
   `Q(A)`, nor store that information in `activation_state`. The prohibited
   behavior is output supervision of `Q(A)`, not every offline weight objective
   that happens to form `A@W`.
2. Produce legal HiF4 fields and keep the API, state, shape, dtype, and device
   behavior valid.
3. Keep the final official evaluation strictly below `420s` (7 minutes).
4. Do not tune from holdout or official-score feedback.

There are no fixed gain, coverage, beam, per-component non-regression, or
intermediate runtime gates beyond these rules. Diagnostic development runs may
exceed 420 seconds. Once an accuracy signal is found, optimize the algorithm
and implementation to fit the final runtime limit.

## Current algorithm

### Linear

1. Reconstruct the floating-point reference from NVFP4 scales and E2M1 data.
2. Search SmoothQuant diagonal scaling, channel permutations, and small
   4/8/16-dimensional orthogonal transforms.
3. Build standard HiF4 parameters and apply 4/8/16-group quadratic refinement.
4. Weight FULL64 uses the legal activation Hessian:
   - retain four scale-beam candidates;
   - process only wide FFN `fc/proj` layers at `0.25` coverage;
   - run GPTQ initialization, one 64-dimensional coordinate sweep, and
     hierarchy toggles;
   - omit the redundant second coordinate sweep.
5. C40 adjacent-128 Block-LDLQ conditional re-solving is retained only in the
   historical archive; the current root does not enable this rejected path.
6. Current C69 sets the dynamic activation quadratic Gram-8 coverage cap to
   `12%`, while retaining sample-local HiF4 encoding and the validated 4/8-group
   refinements.

### Attention

The active A1 path uses Smooth-QK, K midrange centering, headwise permutation,
MHA/GQA alignment, and real-Attention dual-mask safety selection. Fixed H64,
Segment-CVaR, and non-beneficial V-importance candidates remain disabled.

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
docs/real-model-evaluator.md        evaluator usage guide
docs/research/                      literature survey
docs/superpowers/plans/             currently active general workflows
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
missing, run the capture step below first. The main ranking field is
`official_ranking_audit.primary_panel_score_total`; other models are soft
guardrails, while `official_flow_total` is retained for diagnostics.

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

Full test suite:

```powershell
.\.venv\Scripts\python -m pytest -q
```

The full suite also includes real-corpus windows and historical algorithm
regressions. Missing optional dependencies such as `transformers`, or legacy
assertions that intentionally track an older experiment switch, are reported
separately. Before a release, run the syntax checks and the evaluator regression
with a repository-local ignored temp directory:

```powershell
.\.venv\Scripts\python -m pytest -q tests/test_real_model_suite.py --basetemp=.tmp_pytest\readme-verify
```

### Candidate testing order and result capture

Modify only the root `solution.py` for each experiment. Run syntax, compliance,
and single-model real-path checks before multi-model comparison. Do not edit
historical sources under `solutions/`.

1. **Fast pre-commit checks**

   ```powershell
   git diff --check
   .\.venv\Scripts\python -m py_compile solution.py evaluator\real_model_suite.py evaluator\reference_hif4.py evaluator\linear_compliance_guard.py
   .\.venv\Scripts\python -m pytest -q
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

   For full candidate comparisons, the primary score is
   `250 * Linear_mean + 200 * Attention_mean` on the Qwen-shaped panel. The
   smoke command above explicitly uses gpt2-small; local scores are only for
   paired A/B ranking and must not be entered as Official Score. Pair promotion
   runs with `--candidates c39 c41b c47b c66`, and record the full command,
   source/target case counts, total API time, and source SHA256.
   `official_flow_total` remains in JSON for rollback comparisons.

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

5. **Check whether local ordering reproduces official ordering**

   Official anchors are used only for Spearman and pairwise ranking audits;
   the evaluator does not fit absolute official scores. Candidate promotion
   compares Qwen's `primary_panel_score_total` on the same frozen corpus;
   guardrail means, native sums, and Pearson are diagnostics and cannot
   override the primary ordering. Use `--candidates c39 c41b c47b c66` beside
   the revised official anchors.

   The official-flow proxy is:

   ```text
   score(case) = (MSE_STD - MSE_PLAYER) / MSE_STD
   native official_flow_total = sum(all native Linear case scores) + sum(all native Attention case scores)
   qwen panel_score.total = 250 * mean(Linear case scores) + 200 * mean(Attention case scores)
   ```

   Standard NVFP4/HiF4 dequantization, HiF4 parameter validation, and state
   validation are evaluator-owned. A candidate only needs the six official
   APIs. Evaluator-side `A@W` is formed only after the candidate has returned
   its quantized result and is never passed back as calibration data. A
   candidate may form its own `A@W` inside offline
   `hif4_calibration_and_quantize_weight` for `Q(W)` optimization, but it must
   not route that result into `activation_state` or online `Q(A)` selection.

   The task document does not include the source of the official "standard
   HiF4 quantizer." The current independent codec is the historically audited
   implementation and its SHA256 is recorded in every report. Replace it
   bit-for-bit and bump the protocol version when the official function is
   available.

6. **Check the runtime constraint**

   The primary model's complete six-API run must have
   `official_api_total_seconds` strictly below the official `420s` limit;
   exactly 420 seconds fails. Other model proxies are soft guardrails and are
   not added to the submission runtime. Cache reads remove model-forward time
   only and cannot hide a slow candidate. Confirm final end-to-end time with
   the official evaluator.

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
   - Local official-flow Linear sum / cases: ...
   - Local official-flow Attention sum / cases: ...
   - Local official-flow total and paired ordering: ...
   - Local official API total runtime: ...
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
   .\.venv\Scripts\python -m pytest -q
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
- The latest execution history is in
  [2026-08-26-optimization-execution-log.md](logs/execution/2026-08-26-optimization-execution-log.md).
- Candidate archiving follows
  [2026-08-26-solution-archive-workflow.md](docs/superpowers/plans/2026-08-26-solution-archive-workflow.md).
- Multi-model real data, cache modes, and compliance boundaries are documented
  in [real-model-evaluator.md](docs/real-model-evaluator.md).
- Official per-case summation, independent codec/validation, and ranking audit
  are documented in [real-model-evaluator.md](docs/real-model-evaluator.md).
- Superseded optimization plans were moved to
  `docs/superpowers/archive/plans/` and are not active instructions.
