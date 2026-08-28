# HiF4 Quantization Competition Workspace

Development workspace for the Huawei 2026 NVFP4-to-HiF4 algorithm track.
The root `solution.py` is the only active submission file. Historical
candidates live under `solutions/` and are never imported at runtime.

Chinese version: [README.md](README.md)

## Current status

- Latest compliant official champion: v031 / C39-FW, `14613 / 159.2s`.
  v025 / C21-C (`14437 / 166.6s`) remains a secondary anchor.
- Historical v024 scored `16043 / 173.8s`, but it contains Linear output-
  supervision paths disallowed by the later official clarification and is not
  a compliant parent for new work.
- The current root `solution.py` is the frozen C40 robust Block-LDLQ candidate:
  local Linear `0.5393`, causal Attention `0.4497`, and official
  `14432 / 216.667s`. It lost 181 points and added 57.467 seconds versus C39,
  so it is rejected and must not be used as the next parent.
- Current source SHA256:
  `D24BC94F513907CBE97B43865973D1498133D8B9264FAF12661836FF65AAB656`.
- The local evaluator is no longer considered reliable for ranking compliant
  candidates. Both dev and frozen holdout repeat text across calibration and
  test. See the
  [C40 official-result evaluator diagnosis](docs/C40-official-evaluator-diagnosis.md).

Local time and scores are for paired candidate comparison only and are never
reported as official results. Every official result must be archived together
with the exact submitted SHA, score, and runtime.

## Hard competition constraints

1. Never compute `A@W`, directly or indirectly, and use its output to fit,
   select, or infer `Q(A)`.
2. Produce legal HiF4 fields and keep the API, state, shape, dtype, and device
   behavior valid.
3. Keep the final official evaluation strictly below `300s`.
4. Do not tune from holdout or official-score feedback.

There are no fixed gain, coverage, beam, per-component non-regression, or
intermediate runtime gates beyond these rules. Diagnostic development runs may
exceed 300 seconds. Once an accuracy signal is found, optimize the algorithm
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
5. The current root also enables C40 adjacent-128 Block-LDLQ conditional
   re-solving. Its official result failed, so it is retained only for archived
   reproduction and does not represent the champion algorithm.
6. The dynamic Activation path uses sample-local HiF4 encoding and the current
   validated 4/8-group refinements.

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
evaluator/real_data_eval.py         paired real-GPT evaluation
evaluator/synthetic_attention_eval.py
                                    576-case Attention safety matrix
evaluator/real_model_suite.py       multi-model real-data evaluation and cache
evaluator/official_score_calibration.py
                                    frozen official-score fitting and prediction
evaluator/cap_oracle.py             fixed-frame error-space diagnostics
evaluator/linear_compliance_guard.py
                                    static/runtime Linear compliance checks
evaluator/holdout_eval.py           budget-protected holdout evaluation
tests/                              release, format, compliance, algorithm tests
solutions/                          immutable candidate archives
artifacts/real_model_suite/cache/   local real-model snapshots, not committed
docs/superpowers/logs/              execution and calibration records
docs/superpowers/plans/             currently active general workflows
docs/superpowers/archive/plans/     superseded plans, historical only
```

## Running evaluations

Create the project environment:

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install -r evaluator\requirements.txt
```

Real-model evaluation:

```powershell
.\.venv\Scripts\python evaluator\real_data_eval.py `
  --solution solution.py --model models/gpt2 --device cuda
```

GQA example:

```powershell
.\.venv\Scripts\python evaluator\real_data_eval.py `
  --solution solution.py --model models/gpt2 --device cuda --kv-heads 6
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
   .\.venv\Scripts\python -m py_compile solution.py evaluator\real_data_eval.py evaluator\real_model_suite.py
   .\.venv\Scripts\python -m pytest -q
   ```

2. **Test the active root `solution.py`**

   This uses a real GPT-2 forward pass and the deployed candidate API to check
   output format, Linear, Attention, and local runtime:

   ```powershell
   .\.venv\Scripts\python -u evaluator\real_data_eval.py `
     --solution solution.py --model models\gpt2 --device cuda `
     --layers 12 --seq 128 --calib 2 --test 2 `
     --mode amax6 --attn-mask causal
   ```

   Local scores are for paired A/B comparison only and must not be entered as
   Official Score. Record the full command, every Linear component, causal
   Attention, runtime, and the source SHA256.

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
     --report docs\real-model-evaluator-cache-capture-YYYYMMDD.md
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
     --solution solution.py --candidate-name active `
     --device cpu --algorithm-device cuda --cache-mode read `
     --seq 128 --calib 2 --test 4 `
     --output artifacts\real_model_suite\active-YYYYMMDD.json `
     --report docs\real-model-evaluator-active-YYYYMMDD.md
   ```

   In `read` mode, a missing cache, version/configuration mismatch, leaked
   window, or invalid tensor shape fails immediately; it never silently loads
   a model. `auto` is convenient for daily use: it reads a valid cache and
   recaptures a missing or stale one. `write` always refreshes; `off` neither
   reads nor writes. Changing sequence length, calibration/test counts, layer
   cap, model, or the pinned dataset revision requires a new cache.

5. **Predict the official score with a frozen calibration**

   First generate a versioned calibration from the fixed official-anchor
   matrix. Do not refit when the existing v0 is applicable:

   ```powershell
   .\.venv\Scripts\python -u evaluator\official_score_calibration.py fit `
     --input artifacts\real_model_suite\20260828_full.json `
     --output artifacts\real_model_suite\official_score_calibration_v0.json `
     --feature linear_macro_gain

   .\.venv\Scripts\python -u evaluator\official_score_calibration.py predict `
     --calibration artifacts\real_model_suite\official_score_calibration_v0.json `
     --input artifacts\real_model_suite\active-YYYYMMDD.json `
     --output artifacts\real_model_suite\active-YYYYMMDD.official-prediction.json
   ```

   The current v0 has four official anchors and status `diagnostic`. Archive
   every prediction together with leave-one-out MAE, extrapolation status, and
   all five per-model values. Never enter a prediction as Official Score. See
   [official-score-calibration.md](docs/official-score-calibration.md) for the
   complete contract and interpretation.

6. **Check the runtime constraint**

   `algorithm_stage_seconds` must be strictly below the official hard limit of
   `300s`. Cache reads remove model-forward time only; they cannot hide a slow
   candidate algorithm. Confirm the final end-to-end time with the official
   evaluator.

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
   - Local Linear q/k/v/o/fc/proj: ...
   - Local Attention causal: ...
   - Local runtime: ...
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
     docs\superpowers\logs\YYYYMMDD-experiment.md
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
  [2026-08-26-optimization-execution-log.md](docs/superpowers/logs/2026-08-26-optimization-execution-log.md).
- Candidate archiving follows
  [2026-08-26-solution-archive-workflow.md](docs/superpowers/plans/2026-08-26-solution-archive-workflow.md).
- Multi-model real data, cache modes, and compliance boundaries are documented
  in [real-model-evaluator.md](docs/real-model-evaluator.md).
- Frozen fitting and prediction from local metrics to official scores are
  documented in [official-score-calibration.md](docs/official-score-calibration.md).
- Superseded optimization plans were moved to
  `docs/superpowers/archive/plans/` and are not active instructions.
