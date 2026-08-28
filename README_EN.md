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
evaluator/cap_oracle.py             fixed-frame error-space diagnostics
evaluator/linear_compliance_guard.py
                                    static/runtime Linear compliance checks
evaluator/holdout_eval.py           budget-protected holdout evaluation
tests/                              release, format, compliance, algorithm tests
solutions/                          immutable candidate archives
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

## Records

- Current facts come from root `solution.py`, the latest execution log, and
  reproducible evaluator output.
- Historical versions and decisions are indexed in
  [solutions/README.md](solutions/README.md).
- The latest execution history is in
  [2026-08-26-optimization-execution-log.md](docs/superpowers/logs/2026-08-26-optimization-execution-log.md).
- Candidate archiving follows
  [2026-08-26-solution-archive-workflow.md](docs/superpowers/plans/2026-08-26-solution-archive-workflow.md).
- Superseded optimization plans were moved to
  `docs/superpowers/archive/plans/` and are not active instructions.
