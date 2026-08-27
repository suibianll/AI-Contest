# HiF4 Quantization Competition Project

Development workspace for the Huawei 2026 algorithm competition track
(NVFP4 → HiF4). The input is NVFP4 data (E2M1 carrier + block scale);
the output is a HiF4 representation. The goal is to make the dequantized
result as close as possible to the NVFP4 reference while keeping the
compute format legal. Evaluation covers both Linear layers and the
Attention projection path; the score is the MSE improvement relative to
the standard HiF4 conversion.

The official B0 baseline is `youxilee/hif4` v2.0, officially closed at
`15313 / 137s` (user-confirmed 2026-08-27). The current official record
is v013 (C10, `15799 / 144s`). The root `solution.py` is the local
champion (C21) accumulated on top of B0 and validated on the local fixed
matrix; it has no official result yet — local gains must not be
converted into official-score claims.

## Project Structure

```text
solution.py                         The only active, submission-ready algorithm file
evaluator/
  nvfp4_sim.py                      Authoritative NVFP4 encode/decode simulator
  real_data_eval.py                 Real GPT-2 evaluator (defaults to models/gpt2)
  synthetic_attention_eval.py       E1 synthetic attention safety evaluator
  requirements.txt                  Evaluation dependencies
models/gpt2/                        Local GPT-2 weights (~525MB, git-ignored)
solutions/
  README.md                         Master table of versions, scores, runtimes
  YYYYMMDD_vNNN_.../solution.py     Immutable archived algorithm sources
  YYYYMMDD_vNNN_.../result.md       Origin, results, conclusions per version
tests/test_release_candidate.py     Release-candidate checks (incl. E1 subset)
artifacts/                          Raw evidence outputs of local runs
docs/superpowers/                   Design specs, plans, execution logs
```

`solution.py` is the only active file. `solutions/` holds versions that
were submitted or explicitly recorded; it is never a runtime dependency.
`docs/superpowers/` preserves the full design process and is not part of
the competition submission.

## Algorithm Overview

The current v2.0-line solution pipeline:

1. Reconstruct the floating-point reference using the official NVFP4
   scale rules and the E2M1 carrier.
2. Collect calibration activations per Linear layer and search
   SmoothQuant scaling, channel permutations, and weight/activation
   error importance; wide layers use a finer alpha grid.
3. When calibration passes the safety gate, try block-diagonal
   Hadamard transforms, enumerating block sizes 4/8/16 with
   deterministic sign seeds; fall back to the diagonal path if the
   improvement threshold is not met.
4. Apply budgeted hierarchical scale refinement to weights, activations,
   and high-error HiF4 blocks of Q/K/V, using absolute-error ordering,
   quadratic statistics, and boundary extension to increase the gain.
5. Calibrate the attention path on real Q/K/V tensors with MHA/GQA head
   grouping and produce submittable HiF4 parameters through the same
   dynamic-quantization API.

All state lives in plain CPU tensors/scalars. The dynamic-quantization
stage only depends on the calibration state, never on evaluator internals.

## Evaluation Method

The evaluator loads real GPT-2 weights and text-forward activations and
captures per-layer Q/K/V, attention projections, and FFN inputs. For
every sample it produces:

- the NVFP4 dequantized result (reference),
- the standard HiF4 result (baseline),
- the current `solution.py` calibration/dynamic-quantization result
  (candidate).

Each Linear or attention sample is scored with the same formula:

```text
score = (MSE(standard, reference) - MSE(candidate, reference))
        / MSE(standard, reference)
```

then averaged over layers and test batches. The evaluator shares the
core scoring path with the remote hif4 project but adds configurable
`--solution` and `--model` options.

In addition, `evaluator/synthetic_attention_eval.py` (E1) runs a frozen
8-scenario / 576-case synthetic attention matrix (saturated logits,
near-uniform, V outliers, heavy tails, ...) as a safety gate for
attention-path changes — it pre-screens regressions that real-data
evaluation alone cannot expose.

## Environment and Usage

Use the project virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install -r evaluator\requirements.txt
```

GPT-2 weights live in `models/gpt2/` (~525MB, excluded by `.gitignore`);
the evaluator loads this directory by default, no network needed.
Default run (GPT-2, 12 layers, 2 calibration + 2 test batches):

```powershell
.\.venv\Scripts\python evaluator\real_data_eval.py
```

GPU acceleration: `--device cuda` (default `cpu`); `--model` accepts a
Hugging Face name or another local model directory:

```powershell
.\.venv\Scripts\python evaluator\real_data_eval.py `
  --solution solution.py --model gpt2 --device cuda
```

Fast directional comparison during development:

```powershell
.\.venv\Scripts\python evaluator\real_data_eval.py `
  --layers 1 --seq 16 --calib 1 --test 1
```

Synthetic attention safety matrix (all 576 frozen cases):

```powershell
.\.venv\Scripts\python evaluator\synthetic_attention_eval.py `
  --solution solution.py
```

Release checks (state legality, param fields, feature-off equivalence,
synthetic subset):

```powershell
.\.venv\Scripts\python -m pytest tests\test_release_candidate.py -q
```

Key options: `--layers`, `--seq`, `--calib`/`--test`, `--mode`
(`amax6`/`amax4`/`pow2`), `--kv-heads` (GQA smoke runs), `--token-offset`
(pinned local test windows). Output includes the six Linear component
scores (q/k/v/o/fc/proj), causal/non-causal attention scores, and
uniformly bounded algorithm-stage/API timings.

## Local Evaluation and Archival Workflow

The official evaluator is not continuously reachable; the known B0
official result serves as the baseline anchor. Later candidates are
promoted on reproducible local paired results — we do not wait for new
official scores, nor infer official absolute scores from local metrics:

1. B0 and candidates must run paired with identical model, device, mask,
   mode, token offset, and batch counts.
2. Offset `0` is the development set; `97`/`193`/`389` are pinned local
   regression windows (already consumed in the A1 arbitration — no
   tuning against them, and they are no longer claimed as blind sets).
3. Development screening covers `amax6/amax4/pow2`, MHA/GQA, and
   causal/non-causal; head_dim 128 and saturated-logit regimes are
   covered by the frozen synthetic safety matrix.
4. Promotion requires all of: target mean, per-layer tails, state
   legality, E1 synthetic safety track, and the CPU time gate.
5. On promotion, create a local result archive recording the exact
   source SHA256, full configuration, component scores, and timings;
   `Official Score/Time` stays `NA` — never fill in local estimates.

Version history:

- v000: legacy v9 baseline, ~9000+ official;
- v001: former active baseline, `10250 / 127s`;
- v002: `youxilee/hif4` v2.0, official B0, `15313 / 137s`, closed;
- v013: C10 wide-activation quadratic, official record `15799 / 144s`;
- root `solution.py`: local champion (C21 lineage, no official score);
  see `solutions/README.md` and the progressive candidate ledger for
  the current champion ID, source SHA, and fixed-matrix numbers.

A four-anchor calibration of local metrics against official scores
(`docs/superpowers/logs/2026-08-27-evaluator-calibration-report.md`)
shows ~297 official points per 1pp of local Linear mean (stable) but a
weak conversion for local attention gains — local Linear is the
high-leverage metric, while the synthetic matrix is a safety track, not
a scoring lever.

The `.venv/` directory, Python caches, and other local artifacts are
excluded by `.gitignore` and never enter algorithm archives or
competition submissions.
