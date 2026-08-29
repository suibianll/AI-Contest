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
- The current root `solution.py` is C69's activation quadratic Gram-8 candidate;
  its five-model proxy total is `1044.706838`.
- Current source SHA256:
  `1F71CA11FA9707EB9720438EC6D780CC6F520FBA80437B3215398608D5866CA1`.
- The local evaluator is no longer considered reliable for ranking compliant
  candidates. Both dev and frozen holdout repeat text across calibration and
  test. See the
  [C40 official-result evaluator diagnosis](logs/candidates/C40-official-evaluator-diagnosis.md).

Local time and scores are for paired candidate comparison only and are never
reported as official results. Every official result must be archived together
with the exact submitted SHA, score, and runtime.

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
evaluator/real_model_suite.py       multi-model real-data evaluation, cache, and official-flow primary ranking
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

Single-model quick evaluation (gpt2-small, cache-first):

```powershell
.\.venv\Scripts\python -u evaluator\real_model_suite.py `
  --models gpt2-small --solution solution.py --candidate-name active `
  --device cpu --algorithm-device cuda --cache-mode auto `
  --seq 128 --calib 2 --test 4 `
  --output artifacts\real_model_suite\quick-YYYYMMDD.json `
  --report logs\evaluations\quick-YYYYMMDD.md
```

GQA example (Qwen2.5-0.5B ships 14Q/2KV heads with RoPE):

```powershell
.\.venv\Scripts\python -u evaluator\real_model_suite.py `
  --models qwen2.5-0.5b --solution solution.py --candidate-name active `
  --device cpu --algorithm-device cuda --cache-mode auto `
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

### Candidate testing order and result capture

Modify only the root `solution.py` for each experiment. Run syntax, compliance,
and single-model real-path checks before multi-model comparison. Do not edit
historical sources under `solutions/`.

1. **Fast pre-commit checks**

   ```powershell
   git diff --check
   .\.venv\Scripts\python -m py_compile solution.py evaluator\solution_runtime.py evaluator\real_model_suite.py
   .\.venv\Scripts\python -m pytest -q
   ```

2. **Test the active root `solution.py`**

   This uses the cached real-model panel and the deployed candidate API to
   check output format, Linear, Attention, and local runtime:

   ```powershell
   .\.venv\Scripts\python -u evaluator\real_model_suite.py `
     --models gpt2-small --candidates c39 `
     --solution solution.py --candidate-name active `
     --device cpu --algorithm-device cuda --cache-mode read `
     --seq 128 --calib 2 --test 4 `
     --output artifacts\real_model_suite\active-YYYYMMDD.json `
     --report logs\evaluations\active-YYYYMMDD.md
   ```

   The primary ranking score follows the official flow: each test case first
   computes `(MSE_STD-MSE_PLAYER)/MSE_STD`, then all Linear and Attention
   case scores are summed into `official_flow_total`. Always pair with
   `--candidates c39`; local scores are only for paired A/B ranking and must
   not be entered as Official Score. Record the full command, both case
   counts, total API time, and the source SHA256.

3. **Capture real multi-model forward data once**

   `real_model_suite.py` covers GPT-2 small/medium, OPT-125M, Pythia-160M,
   and Qwen2.5-0.5B by default, and compares the registered C21/C38/C39/C40
   anchors. Capture model data first so each candidate does not repeat model
   forward execution:

   ```powershell
   .\.venv\Scripts\python -u evaluator\real_model_suite.py `
     --device cuda --algorithm-device cuda --cache-mode write --capture-only `
     --seq 128 --calib 2 --test 4 `
     --output artifacts\real_model_suite\cache-capture-YYYYMMDD.json `
     --report logs\evaluations\cache-capture-YYYYMMDD.md
   ```

   Replace `YYYYMMDD` with the actual run date. Snapshots are stored in
   `artifacts/real_model_suite/cache/` and are not committed to Git. They
   contain real model weights, Linear inputs, real Q/K/V, token ids, model/data
   revisions, and window-validation metadata.

4. **Evaluate from cache only**

   After capture, candidate tests do not load a tokenizer/model, execute model
   forward, or access the network:

   ```powershell
   .\.venv\Scripts\python -u evaluator\real_model_suite.py `
     --candidates c39 --solution solution.py --candidate-name active `
     --device cpu --algorithm-device cuda --cache-mode read `
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
   compares `official_flow_total` on the same frozen case panel. `global_gain`,
   component means, and Pearson are diagnostics and cannot override the
   primary ordering. Always use `--candidates c39` to run active beside the
   current official Champion.

   The official-flow proxy is:

   ```text
   score(case) = (MSE_STD - MSE_PLAYER) / MSE_STD
   official_flow_total = sum(all Linear case scores) + sum(all Attention case scores)
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

   Each complete six-API run on a model proxy must have
   `official_api_total_seconds` strictly below the official `420s` limit;
   exactly 420 seconds fails. The multi-model suite uses the maximum proxy
   time for this check and does not add independent proxy runtimes to pretend
   they are one official submission. Cache reads remove model-forward time
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
