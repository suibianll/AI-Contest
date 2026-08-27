# HiF4 Quantization Competition Project

Development workspace for the Huawei 2026 algorithm competition track
(NVFP4 → HiF4). The input is NVFP4 data (E2M1 payload plus block scale), and
the output is a HiF4 representation. The goal is to make the dequantized result
as close as possible to the NVFP4 reference while keeping the compute format
legal. Evaluation covers both Linear layers and the Attention projection path;
the score is the MSE improvement relative to standard HiF4 conversion.

The official B0 baseline is `youxilee/hif4` v2.0, officially closed at
`15313 / 137s` (user-confirmed on 2026-08-27). The current official record is
v024 (C21, `16043 / 173.8s`, user-confirmed on 2026-08-27); the root
`solution.py` is byte-identical to that archive. Local gains must not be
converted into official-score claims.

Chinese version: [README.md](README.md)

## Project Structure

```text
solution.py                         The only active, submission-ready algorithm file
evaluator/
  nvfp4_sim.py                      Authoritative NVFP4 encode/decode simulator
  real_data_eval.py                 Real GPT-2 evaluator (defaults to models/gpt2)
  synthetic_attention_eval.py       E1 synthetic Attention safety evaluator
  requirements.txt                  Evaluation dependencies
models/gpt2/                        Local GPT-2 weights (~525MB, git-ignored)
solutions/
  README.md                         Master table of versions, scores, and runtimes
  YYYYMMDD_vNNN_.../solution.py     Immutable archived algorithm sources
  YYYYMMDD_vNNN_.../result.md       Origin, results, and conclusions per version
tests/test_release_candidate.py     Release-candidate checks (including E1 subset)
artifacts/                          Raw evidence outputs from local runs
docs/superpowers/                   Design specs, plans, and execution logs
```

`solution.py` is the only active file. `solutions/` stores versions that were
submitted or explicitly recorded and is never a runtime dependency.
`docs/superpowers/` preserves the complete design process and is not part of
the competition submission.

## Algorithm Overview

The current v2.0-line solution pipeline:

1. Reconstruct the floating-point reference using the official NVFP4 scale
   rules and the E2M1 payload.
2. Collect calibration activations for each Linear layer and search
   SmoothQuant scales, channel permutations, and Weight/Activation error
   importance; wide layers use a finer alpha grid.
3. When calibration passes the safety gate, try deterministic signed Hadamard
   block-diagonal transforms of sizes 4/8/16; fall back to the diagonal path if
   the improvement threshold is not met.
4. Apply budgeted hierarchical scale refinement to high-error HiF4 blocks in
   Weights, Activations, and Q/K/V, using absolute-error ordering, quadratic
   statistics, and boundary extension to improve the gain.
5. Calibrate the Attention path on real Q/K/V tensors with MHA/GQA head
   grouping and produce submittable HiF4 parameters through the same dynamic
   quantization API.

All state is stored in plain CPU tensors or scalars. Dynamic quantization only
depends on calibration state and never on evaluator internals.

## How the Current Pipeline Actually Runs

This section follows the real code path of the current `solution.py`
(C21/v024). Function names, execution order, and constants correspond directly
to the source.

### 1. Evaluator and Six Official API Calls

`evaluator/real_data_eval.py` runs GPT-2 forward passes for two calibration
batches and two test batches. Hooks capture per-layer activations, after which
the evaluator calls the solution in this order:

```text
Calibration stage (once per layer)
  hif4_calibration_and_quantize_weight(weight_quant, weight_scale, calib_act_list)
      → {weight_params(five fields), activation_state}
  hif4_calibration_attention(calib_qkv_list, q_heads, kv_heads, head_dim)
      → {q_state, k_state, v_state}

Test stage (once per layer and batch)
  hif4_dynamic_quantize_activation(act_quant, act_scale, activation_state) → five fields
  hif4_dynamic_quantize_q / _k / _v(...) → five fields
```

Linear and Attention use the same scoring formula:

```text
score = (MSE(standard, reference) - MSE(candidate, reference))
        / MSE(standard, reference)
```

Here, standard is naive HiF4 (amax/7, threshold-based lv2/lv3, rounded
mantissa, and no refinement), candidate is the dynamic quantization result of
this solution, and reference is the dequantized NVFP4 floating-point value.

### 2. HiF4 Target Format (Five Legal Fields)

The input is NVFP4 (an E2M1 payload plus one scale per 16-channel block).
`_dequantize_nvfp4_float32` first reconstructs a dense float32 tensor. The
output treats the final 64 channels as one top-level block and decomposes it
hierarchically:

```text
x = sign * mant * scale_lv3 * scale_lv2 * scale_factor

[... , 64] → reshape [..., blocks, 8, 2, 4]
scale_factor  One per top-level 64 block; E6M2 floating scale
              (standard uses the nearest E6M2 code to amax/7)
scale_lv2     One per 8-channel group; value in {1, 2} (group-level ×2 exponent)
scale_lv3     One per 4-channel subgroup; value in {1, 2}
sign          ±1 (canonicalized to 0 when mantissa=0)
mant          Mantissa in {0, 0.25, ..., 1.75}
              (code×0.25, code in 0..7)
```

The maximum representable inner value is
`1.75 × 2 × 2 = 7 × scale` (`_HIF4_MAX_INNER = 7.0`). Dequantization multiplies
the five fields element by element in `_dequantize_hif4`.

### 3. Weight Calibration (`hif4_calibration_and_quantize_weight`)

Each Linear layer executes the following sequence:

1. **Statistics collection**: dequantize calibration activations and accumulate
   the per-channel second moment `sum_square`, amax, and, when
   `_WEIGHT_QUADRATIC` is enabled, the full covariance
   `cov_sum = X^T X`.
2. **SmoothQuant candidates**: use
   `d = act_amax^α / w_amax^(1-α)` with alpha grid
   `(0.25, 0.5, 0.75)`; wide layers (input or output at least 2048) use a
   five-point grid. Each alpha produces both amax and RMS variants, normalized
   by their geometric mean to prevent global drift.
3. **Channel permutation candidates**: `_hierarchy_aware_permutation` places
   channels of similar magnitude in the same 64 block to make within-block
   amax values more uniform and improve scale utilization. It then adds four
   one-sided range orderings based on Weight/Activation amax and RMS.
4. **Candidate scoring and gating**: each `(d, perm)` candidate is scored on
   sampled rows (at most 256 Weight rows and 128 activation-token samples).
   `_candidate_is_safe` requires sufficient mean improvement without
   worst-sample degradation; otherwise the candidate is rejected.
5. **Block Hadamard transform** (a Matrix SmoothQuant extension): after choosing
   `d/perm`, try signed Hadamard transforms of sizes 4/8/16. The transforms are
   orthogonal and state stores only the two integers `block_smooth_size/seed`.
   This step is scored by `_linear_output_candidate_metrics`, a real Linear
   output oracle, and is a known violation under the official `A @ W` ban.
6. **One-time full transform**: apply the winning combination to the complete
   Weight as `W_t = W · D · P · R`. It pairs exactly with the Activation-side
   transform `X_t = X · D⁻¹ · P · R`, algebraically preserving
   `X_t · W_t^T = X · W^T`.
7. **Weight encoding**: call `_dense_to_hif4(weight_t, importance=diagonal H_x,
   gram=4×4 block-diagonal covariance, search_offsets, budget)`, as detailed in
   Section 4.
8. **8/16-group second-order refinement**: `_refine_weight_groups8/16`
   performs coordinate-level refinement, using the incremental `H·e` formula,
   on top-K high-loss groups of 8 or 16 channels.
9. **Build activation_state**: store `smooth_inv = 1/d`, `permutation`,
   `block_smooth_*`, `importance` from the column energy of `weight_hat`,
   `gram/gram8/cross8` (Weight-space Gram and cross term, with the latter being
   non-compliant), offset sets, and refinement budgets. Data-driven mode sets
   the ratio from the fraction of loss captured.

### 4. Core Encoder (`_dense_to_hif4`)

All tensors share this quantization path:

1. Reshape to `[blocks, 8, 2, 4]` and extract sign and absolute value.
2. Encode the standard scale `amax/7` to the nearest E6M2 code and decode it
   again to guarantee a legal representation.
3. Apply threshold hierarchy rules:
   `max8 ≥ 4·scale → lv2=2` and
   `max4 ≥ 2·scale·lv2 → lv3=2`; compute mantissa as
   `round(|x|·4/denominator)`, clamp to 0..7, and multiply by 0.25.
4. **Hard-block selection**: blocks with normalized error above `1e-7` enter
   the refinement pool. Absolute weighted loss determines the top-K subset,
   capped by `max_refine_ratio × total blocks` and an absolute block limit.
5. **Batched offset search**: expand `standard_code + offsets` for hard blocks
   along the offset dimension to `[K, N]`, then call
   `_solve_exact_hierarchy` once for an exact solve. For each scale candidate,
   enumerate total exponent `2^e (e=0,1,2)`, create three loss tables using
   diagonal importance or the quadratic form `Δ^T G Δ`, solve
   lv2/lv3/mantissa exactly, and choose the offset by a per-block argmin.
6. **Boundary extension**: if the winning offset lies on the search boundary,
   continue outward for at most two steps. The Weight-side range is
   `(-2,…,3)` and the Activation-side range is `(-1,…,3)`.
7. **L1 data-driven scale**: currently disabled by
   `_L1_DATA_DRIVEN_SCALE = False`. When enabled, generate least-squares scale
   and quantile-trim candidates from the current winner, then apply the same
   exact solve and per-block fallback.
8. **Acceptance gate**: refined parameters are written only when
   `best_loss ≤ (1-margin)·standard_loss`; otherwise the block keeps standard
   parameters. Refinement is only allowed to improve the objective.

### 5. Dynamic Activation Quantization (`hif4_dynamic_quantize_activation`)

Every test batch invokes this path once per layer, entirely without gradients:

```text
Dequantize NVFP4 → × smooth_inv → channel permutation → block Hadamard
→ _dense_to_hif4(importance, gram4, offsets, budget)
→ _refine_weight_groups8(gram8, optional cross8)  # top-K 8-channel refinement
```

Activation-side gram/gram8 comes from the block diagonal of the Weight-space
operator `weight_smooth^T · weight_smooth`; cross8 comes from the cross term
`(W_hat − W)·W_hat^T`. These state fields belong to the C18-C21 cross
mechanism and are pending removal under the official ban.

### 6. Attention Calibration (`hif4_calibration_attention`)

Q/K/V use independent paths. The core idea is a strictly equivalent transform
of the `Q·K^T` dot product: `d_kv` is aligned per head, Q is multiplied by `d`,
and K by `1/d`, preserving the dot product.

1. **Statistics**: compute per-head second moments and peaks; K also receives a
   version after midrange centering.
2. **A1 context**: compute reference Attention from real Q/K/V calibration
   prefixes on both causal and non-causal tracks. Hold V quantization fixed to
   isolate Q/K transform selection.
3. **Smooth-QK**: use `d = k_peak^α / q_peak^(1-α)`, aligned at KV-head
   granularity for GQA.
4. **K centering**: apply exact midrange centering based on softmax translation
   invariance.
5. **Headwise permutation**: Q and K share the same within-head permutation,
   preserving their dot product.
6. **Dual-track selection and final gate**: the A1 track selects transforms by
   real output error; the proxy track selects by the B0-style reconstruction
   proxy. After each produces a winner, rerun output error through the complete
   deployed path `hif4_dynamic_quantize_q/k/v`. Fall back to the proxy winner
   when A1 lacks a clear advantage or the safety track regresses.
7. **A3 V importance**: after Q/K are fixed, compare head-level
   `E[A²]`, `E[A]`, and `E[A²]+E[A]²`, again through the real output gate.
8. Return `q_state/k_state/v_state`, containing multipliers, permutations,
   importance, offsets, and refinement budgets. A2 H64 rotation is disabled by
   default.

Dynamic Q: `× d_q → permutation → encode(importance=h_k, offset search)`;
dynamic K: `center → × 1/d → permutation → encode(importance=h_q)`;
dynamic V: `encode(head-level importance)`.

### 7. Pipeline Summary

```text
               Calibration (once/layer, CPU state)          Dynamic (each batch)
  ┌─────────────────────────────────────────────┐   ┌──────────────────────┐
  │ NVFP4 → float32 → stats(moment/amax/cov)      │   │ NVFP4 → float32      │
  │ → Smooth d / permutation P / Hadamard R       │   │ → ×D⁻¹ → P → R       │
  │   search and gating                           │   │ → _dense_to_hif4     │
  │ → W_t = W·D·P·R (strictly equivalent)         │   │   (offset + hierarchy)│
  │ → _dense_to_hif4 + 8/16-group refinement      │   │ → 8×8-group refine   │
  │ → activation_state {D⁻¹,P,R,importance,       │   │ → five HiF4 fields   │
  │    gram/gram8/cross8, budget}                 │   │                      │
  └─────────────────────────────────────────────┘   └──────────────────────┘
  Attention: Smooth-QK + K centering + headwise permutation
             (strict dot-product invariance), A1 real-output gating
```

Design principle: every equivalent transform relies only on algebraic
identities—paired scaling, paired permutation, or paired orthogonal
rotation—to preserve `X·W^T` and `Q·K^T`. Quantization error is reduced in the
transformed coordinate system. All state is plain CPU tensor data, and the
dynamic stage does not depend on evaluator internals.

Compliance note: Step 5 of Section 3, cross8 in Section 5, and the
output-supervised `_linear_output_candidate_metrics` and
`_activation8_gate_decisions` paths in Weight calibration are pending removal
under the official `A @ W` ban. The remediation is specified in Phase 0 of
`docs/superpowers/plans/2026-08-27-hif4-26000-algorithm-implementation-plan.md`.

## Latest Verified Algorithms

The current official champion is v024 (candidate C21, commit `23d1cf7`):
`16043 / 173.8s` official, `+730` over B0 (v002, `15313 / 137s`) and
`+244` over v013 (`15799 / 144s`). The SHA256 of the root `solution.py` is
`40F4D17C12F976F83856B9641BE9A3951867BC8979992D773C60C0C1C3E8066A`.
Git blob verification confirms that it is byte-identical to the v024 archive,
which contains exactly the bytes evaluated by the official system.

The tables below list the closed official anchors and the verified mechanism
chain of C21. Each mechanism is an individually archived, single-mechanism
candidate; main effects are the offset-0 records from the candidate ledger.

Closed official anchors:

| Version | Mechanism | Official Score | Time |
|---|---|---:|---:|
| v000 | v9 baseline | ~9000+ | NA |
| v001 | former baseline | 10250 | 127s |
| v002 (B0) | youxilee/hif4 v2.0 | 15313 | 137s |
| v013 (C10) | wide-layer Activation quadratic refinement | 15799 | 144s |
| v024 (C21) | gated exact cross selection | 16043 | 173.8s |

C21 mechanism chain:

| # | Mechanism | Candidate | Verification | Main Effect (offset 0) |
|---|---|---|---|---|
| 1 | output-aware Attention selector | C1 / v003 | local | causal Attention +7.12pp |
| 2 | top-K 8×8 Weight quadratic refinement | C3 / v006 | local 6/6 | Linear +1.10pp |
| 3 | top-K 16×16 Weight quadratic refinement | C5 / v008 | local 6/6 | Linear +0.23pp |
| 4 | wide (3072 FFN) Activation quadratic refinement | C10 / v013 | official | proj +0.54pp |
| 5 | wide Activation 8×8 residual | C11 / v014 | local 6/6 | proj +0.31pp |
| 6 | calibration-gated all-width Activation 8×8 | C14 / v017 | local 6/6, all components safe | Linear +0.45pp |
| 7 | gated Activation 8×8 coverage 8% | C17 / v020 | local 6/6, 36/36 components | Linear +0.29pp |
| 8 | calibration-gated exact cross selection | C21 / v024 | official | Linear +0.15pp; fixes C20 pow2 fallback |

Cumulative effect: the Attention path retains A1's `+7.12pp` causal gain;
Linear mean increased from `0.5668` in C1 to `0.5930` in C21, about
`+2.62pp`. Every candidate passed both the fixed regression matrix in
`evaluator/real_data_eval.py` (amax6/amax4/pow2 × MHA/GQA ×
causal/non-causal, offsets 0/97/193/389) and the frozen synthetic matrix in
`evaluator/synthetic_attention_eval.py` (8 scenarios, 576 cases).

Compliance note: the competition has clarified that Linear calibration must
not fit `Q(A)` from `A @ W` or mathematically equivalent output supervision.
C21's Linear calibration contains output-supervised paths such as
`_linear_output_candidate_metrics` and `group_cross8`, making it non-compliant
under the clarified rule. The next mainline, HiF4-OSQ, first removes those
paths to build the compliant baseline C21-C, then adds a 64-dimensional
Hadamard rotation, full-64 GPTQ Weight refinement, top-K full-64 Activation
solving, and a learnable equivalent scale. See
`docs/superpowers/plans/2026-08-27-hif4-26000-algorithm-implementation-plan.md`.
The primary official target is `22000~25000`, `26000` is a stretch target, and
the official time limit is `300s`.

## Evaluation Method

The evaluator loads real GPT-2 Weights and text-forward Activations and
captures per-layer Q/K/V, Attention projections, and FFN inputs. For every
sample it produces:

- the NVFP4 dequantized result (reference);
- the standard HiF4 result (baseline);
- the current `solution.py` calibration/dynamic-quantization result
  (candidate).

Each Linear or Attention sample is scored with the same formula:

```text
score = (MSE(standard, reference) - MSE(candidate, reference))
        / MSE(standard, reference)
```

The scores are then averaged across layers and test batches. The evaluator
shares its core scoring path with the remote hif4 project and adds configurable
`--solution` and `--model` options.

`evaluator/synthetic_attention_eval.py` (E1) runs a frozen 8-scenario,
576-case synthetic Attention matrix covering saturated logits, near-uniform
distributions, V outliers, heavy tails, and related cases. It acts as a safety
gate for Attention-path changes and pre-screens regressions that real-data
evaluation alone may not expose.

## Environment and Usage

Use the project virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install -r evaluator\requirements.txt
```

GPT-2 Weights live in `models/gpt2/` (~525MB, excluded by `.gitignore`). The
evaluator loads this directory by default, with no network access required.
The default run uses GPT-2 with 12 layers, two calibration batches, and two
test batches:

```powershell
.\.venv\Scripts\python evaluator\real_data_eval.py
```

Use `--device cuda` for GPU acceleration; the default is `cpu`. `--model`
accepts a Hugging Face name or another local model directory:

```powershell
.\.venv\Scripts\python evaluator\real_data_eval.py `
  --solution solution.py --model gpt2 --device cuda
```

Fast directional comparison during development:

```powershell
.\.venv\Scripts\python evaluator\real_data_eval.py `
  --layers 1 --seq 16 --calib 1 --test 1
```

Run all 576 frozen synthetic Attention cases:

```powershell
.\.venv\Scripts\python evaluator\synthetic_attention_eval.py `
  --solution solution.py
```

Release checks, including state legality, parameter fields, feature-off
equivalence, and the synthetic subset:

```powershell
.\.venv\Scripts\python -m pytest tests\test_release_candidate.py -q
```

Key options are `--layers`, `--seq`, `--calib`/`--test`, `--mode`
(`amax6`/`amax4`/`pow2`), `--kv-heads` for GQA smoke runs, and
`--token-offset` for pinned local test windows. Output includes the six Linear
component scores (q/k/v/o/fc/proj), causal/non-causal Attention scores, and
uniformly bounded algorithm-stage/API timings.

## Local Evaluation and Archival Workflow

The official evaluator is not continuously reachable, so the known B0
official result serves as the baseline anchor. Later candidates are promoted
on reproducible local paired results. We do not wait for new official scores
or infer official absolute scores from local metrics:

1. B0 and candidates must be evaluated as a pair with identical model, device,
   mask, mode, token offset, and batch counts.
2. Offset `0` is the development set. Offsets `97`, `193`, and `389` are pinned
   local regression windows. They were already consumed in A1 arbitration,
   are no longer claimed as blind sets, and must not be used for tuning.
3. Development screening covers `amax6/amax4/pow2`, MHA/GQA, and
   causal/non-causal. Head dimension 128 and saturated-logit regimes are
   covered by the frozen synthetic safety matrix.
4. Promotion requires all of the target mean, per-layer tails, state legality,
   the E1 synthetic safety track, and the CPU time gate.
5. After promotion, create a local result archive containing the exact source
   SHA256, complete configuration, component scores, and runtimes.
   `Official Score/Time` remains `NA`; never fill it with a local estimate.

Version history:

- v000: legacy v9 baseline, approximately 9000+ official;
- v001: former active baseline, `10250 / 127s`;
- v002: `youxilee/hif4` v2.0, official B0, `15313 / 137s`, closed;
- v013: C10 wide-layer Activation quadratic refinement,
  `15799 / 144s` official;
- v024: C21 gated exact cross selection, current official record
  `16043 / 173.8s`; the root `solution.py` is byte-identical to this archive.
  See `solutions/README.md` and the progressive candidate ledger for the full
  chain, source SHAs, and fixed-matrix results.

A four-anchor calibration of local metrics against official scores is recorded
in `docs/superpowers/logs/2026-08-27-evaluator-calibration-report.md`. A 1pp
increase in local Linear mean corresponds to approximately 297 official points
and has been relatively stable, while the conversion of local Attention gains
is weak. Local Linear is therefore the high-leverage metric, while the
synthetic matrix is a safety track rather than a scoring lever.

The `.venv/` directory, Python caches, and other local artifacts are excluded
by `.gitignore` and never enter algorithm archives or competition submissions.
