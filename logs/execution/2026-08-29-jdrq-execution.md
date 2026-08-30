# JDRQ execution log — 2026-08-29

## C72/C73/C74 fixed-Q(A) weight search

- Parent SHA: current root C69-series `solution.py` before this change (not overwritten)
- C74 archive SHA256: `61C216BEE1ECA9DB6185BCD49C679A34B684F540451CD5562A2008D0DA4B2AD9`
- Active/v073 SHA256: `A0DCE5D79DA931D5B67FACCBA47226B6C8FCE9FC9551200ED86A3693A1E464DA`
- Active change: frozen-state product builder, dual/block ridge research arm, mantissa coordinate descent, bounded E6M2/lv2/lv3 hierarchy beam, source-aware activation scale proposals, and project-only gram64 activation refinement
- Compliance: `activation_state` is created before JDRQ and is not mutated; product, residual, `Z` and candidate scores are calibration-local
- Train/holdout: two calibration windows are retained as separate robust-score folds; evaluator test windows are never passed to `solution.py`
- Default: C74 hierarchy arm on; full dual target arm off pending migration-gap study

### Local evaluator evidence

| model | native total | API time | device | exact observation |
|---|---:|---:|---|---|
| GPT-2 small | 160.571830 | 61.88s | CUDA | current root with C74; root control with JDRQ disabled 155.604260 |
| Qwen2.5-0.5B | 356.605602 | 163.41s | CUDA | C74 improves C66-screen 350.152420 |
| OPT-125M | 85.580941 | 59.56s | CUDA | C74 does not reproduce C71's negative collapse |
| Pythia-160M | 179.059425 | 59.71s | CUDA | near C66 control 178.939 |

### Unit/compliance evidence

```text
python -m pytest -q tests/test_jdrq.py                 # 5 passed
python -m pytest -q tests/test_reference_hif4.py \
    tests/test_jdrq.py tests/test_linear_compliance_guard.py \
    tests/test_linear_error_decomposition.py             # 29 passed
python -m pytest -q tests/test_release_candidate.py \
    -k "not local_holdout_offsets"                      # 20 passed, 1 deselected
```

The full repository pytest still has one pre-existing assertion mismatch in
`tests/test_weight_full64.py::test_weight_full64_wide_only_keeps_narrow_path_equal_to_c21c`
(`max_refine_ratio` differs between the current C69 root and the historical
C21C expectation). The JDRQ/C75 implementation itself passes syntax,
compliance and focused tests.

### Decision

Continue with C75. The hierarchy residual mechanism has a reproducible main-
model signal and stays below the runtime limit on CUDA. The dual target is
retained as a falsifiable research switch because its current dense fit improves
calibration loss without improving held-out loss.

### D0 ceiling snapshots

`evaluator/jdrq_diagnostics.py` on layer-0 down-proj reports:

| model | parent loss | continuous dual | legal projection | hierarchy |
|---|---:|---:|---:|---:|
| GPT-2 small | 0.006693 | 0.000082 | 0.003706 | 0.006481 |
| Qwen2.5-0.5B | 0.002686 | 0.000012 | 0.001883 | 0.002648 |

The continuous ceiling is far below the parent, while the legal projection
retains a 45–70% gap to that ceiling.  This is evidence that the next gain is
primarily a better discrete target/solver (and eventually a better frozen
activation proposal), not another small fixed offset sweep.

## C75 source-aware activation + project-only gram64

### Implemented

- C75.1 adds median/q75/max log-domain E6M2 proposals from the four NVFP4
  E4M3 source scales in each HiF4 64-group. The old amax and hierarchy offset
  candidates stay in the pool, so this is a proposal expansion rather than a
  hard replacement.
- C75.2 builds static per-group `W.T@W` slices during calibration and applies
  one exact signed-mantissa lattice sweep to the highest-loss activation
  blocks. The tensor is bounded and stored as static CPU state only.
- Attribution showed all-shape gram64 could migrate GPT-2 square q/k/v/o
  paths. The active policy is therefore shape-derived: enable gram64 only
  when `out_features < in_features`, with no model-name gate. After the state
  changes, C72--C74 rebuild `Z` and reselect static `Q(W)`.

### Real-model observations

| model | configuration | native total | API time | note |
|---|---|---:|---:|---|
| GPT-2 small | source-aware + project-only gram64 | 158.561896 | 59.64s CUDA | completed; linear 137.255660, attention 21.306236 |
| GPT-2 small | source-aware, no gram64 | 160.597330 | 65.79s CUDA | ablation/source arm |
| GPT-2 small | all-shape gram64 | 159.690510 | 65.79s CUDA | square-path migration noise |
| Qwen2.5-0.5B | all-shape gram64 | 363.937585 | 189.64s CUDA | directional attribution |
| OPT-125M | all-shape gram64 | 87.315586 | 69.60s CUDA | directional attribution |
| Pythia-160M | all-shape gram64 | 182.048492 | 70.71s CUDA | directional attribution |

### Project-only gate rerun

After the shape-derived `out_features < in_features` gate was enabled, all
models were rerun from the finalized activation state and a fresh fixed-state
JDRQ search:

| model | native total | panel total | Linear | Attention | API time |
|---|---:|---:|---:|---:|---:|
| GPT-2 small | 158.561896 | 207.921523 | 137.255660 | 21.306236 | 59.64s |
| Qwen2.5-0.5B | 360.658419 | 242.190891 | 297.538702 | 63.119717 | 168.72s |
| OPT-125M | 85.772835 | 139.265384 | 66.125233 | 19.647602 | 61.14s |
| Pythia-160M | 179.473083 | 289.874153 | 138.825204 | 40.647879 | 61.98s |

Compared with C74, Qwen gains `+4.052817` native points; OPT and Pythia
remain positive and do not reproduce the earlier C71 collapse. All API times
remain below the 420-second limit. The four runs are stored at
`artifacts/real_model_suite/active-c75-proj64-{qwen,hetero}.json` and the
corresponding Markdown reports under `logs/evaluations/`.

Focused command after the C75.2 fixes:

```text
python -m py_compile solution.py evaluator/jdrq_diagnostics.py tests/test_jdrq.py
python -m pytest -q tests/test_jdrq.py tests/test_release_candidate.py \
    -k "not local_holdout_offsets"                      # 25 passed, 1 deselected
```

v073 is archived under `solutions/20260829_v073_c75-source-aware-gram64_scoreNA_timeNA/`.
The next measurement is C75.3 transform-candidate successive halving; no
official-score claim is inferred from the local native/panel values.

## C75.3--C75.6 rowwise JDRQ and wide gram64 (v074)

### Active mechanisms

- Rowwise JDRQ hierarchy gives each output row its own top-2 64-block budget
  when the channel width is at most 4096; Qwen's 4864-wide projection keeps the
  global hierarchy path. A 0.35 soft validation-window mix ranks rowwise and
  global candidates without a hard per-fold veto.
- Wide activation gram64 is enabled only for shape-derived down-projections up
  to 8192 channels. It stores static 64x64 slices of `W.T@W`, applies a
  full-H E6M2/lv2/lv3 beam with offsets `-4..4`, then one signed-mantissa
  sweep. No product output or residual enters activation state.
- H32/H64 remain optional operand-local block-Hadamard candidates. An offline
  output-product reranker was tested but disabled in the release path after the
  compliance runtime audit identified provenance-tainted intermediate products;
  this does not disable JDRQ's legal offline `A@W` static-weight objective.
- The four NVFP4 source E4M3 scales remain a proposal-only input to the online
  activation quantizer. Whenever that state changes, the frozen-Q(A) JDRQ
  search is rerun and the old Q(W) is never reused.

### Real-model observations

| model | native total | panel total | Linear | Attention | API time |
|---|---:|---:|---:|---:|---:|
| GPT-2 small (rowwise2) | 158.550907 | 207.911984 | 137.244671 | 21.306236 | 68.96s |
| OPT-125M (rowwise2) | 85.736733 | 139.234046 | 66.089132 | 19.647602 | 69.03s |
| Pythia-160M (rowwise2) | 179.446007 | 289.850650 | 138.798128 | 40.647879 | 67.30s |
| Qwen2.5-0.5B (wide gram64/hierarchy) | 361.503707 | 242.505358 | 298.383991 | 63.119717 | 179.27s |

The H32/H64 output-reranking experiment produced the same Qwen score as the
wide hierarchy parent (`361.503707`) before it was disabled for compliance.
The native/panel values above are local relative measurements, not official
score conversions; every API time is below the 420-second limit.

### Verification and archive

```text
python -m py_compile solution.py evaluator/jdrq_diagnostics.py tests/test_jdrq.py
python -m pytest -q tests/test_jdrq.py tests/test_linear_compliance_guard.py \
    tests/test_reference_hif4.py tests/test_release_candidate.py \
    -k "not local_holdout_offsets"       # 48 passed, 1 deselected
```

The current root and `solutions/20260829_v074_c75-rowwise-jdrq_scoreNA_timeNA/`
are byte-identical after the final copy; the SHA256 is recorded in the v074
archive result. The next experiment starts from this immutable v074 parent and
targets C76 Attention Q/K structure rather than reopening rejected cross-term
or output-reranker paths.

## C76.4 GQA head-local rotation (v075)

### Implementation

The active C76 arm searches signed orthogonal Hadamard transforms inside each
attention head: H16/H32/H64 block sizes and four deterministic sign seeds.
Q heads sharing one KV head reuse the same transform, preserving the continuous
QK dot product exactly under GQA. The dynamic state carries only CPU `rotation`
signs and `rotation_block`; the transform is selected through the real deployed
Q/K quantizer on calibration prefixes. A structural `q_num_heads !=
kv_num_heads` gate keeps MHA on the v074 path while Qwen-like GQA receives the
new search.

C76.1 independent head permutations, C76.2 output-Fisher importance and C76.3
reciprocal temperature were also implemented as switches. Their Qwen
calibration-to-test runs regressed (`359.273350` when combined and
`359.646532` for Fisher-only), so they remain disabled research arms and are not
part of v075.

### Real-model observations

| model | candidate | native total | panel total | Linear | Attention | API time |
|---|---|---:|---:|---:|---:|---:|
| Qwen2.5-0.5B | c76-rot-gqa-only | 369.344509 | 258.840363 | 298.383991 | 70.960519 | 188.06s |
| GPT-2 small | c76-rot (MHA gate skipped) | 159.373865 | 211.028560 | 137.339381 | 22.034483 | 72.19s |
| OPT-125M | c76-rot (MHA gate skipped) | 85.625086 | 138.872326 | 66.057761 | 19.567325 | 70.30s |
| Pythia-160M | c76-rot (MHA gate skipped) | 180.249839 | 292.741517 | 138.937105 | 41.312734 | 71.91s |

The MHA rows above are from the ungated exploratory run and explain why the
released policy is GQA-only; with the gate they are identical to v074. Qwen's
Attention native component rises from `63.119717` to `70.960519` while the API
time remains well below 420 seconds. Native/panel numbers are local relative
diagnostics, not official score conversions.

### Verification and archive

```text
python -m py_compile solution.py evaluator/jdrq_diagnostics.py tests/test_jdrq.py
python -m pytest -q tests/test_jdrq.py tests/test_linear_compliance_guard.py \
    tests/test_reference_hif4.py tests/test_release_candidate.py \
    -k "not local_holdout_offsets"       # 48 passed, 1 deselected
```

The active root and `solutions/20260830_v075_c76-gqa-rotation_scoreNA_timeNA/`
are byte-identical with SHA256
`DCA23116D76033A7EB5A04C5CC7EF003A52995905261699B2D06883D4C0BE4A4`.

## C77 all-shape gram64 activation refinement (v076)

The earlier project-only gate for activation `gram64` was removed.  The
calibration path now builds static block-diagonal `W.T @ W` slices for every
Linear shape within the existing width caps; the state still contains only a
CPU tensor and the standard dynamic HiF4 fields.  C76.4 GQA rotation remains
unchanged.  This is a shape policy change, not a model-name or role lookup.

| model | native total | panel total | Linear | Attention | API time |
|---|---:|---:|---:|---:|---:|
| Qwen2.5-0.5B | 372.623675 | 260.060290 | 301.663157 | 70.960519 | 207.72s |
| GPT-2 small | 159.774232 | 208.973897 | 138.467995 | 21.306236 | 78.68s |
| OPT-125M | 87.248114 | 140.546008 | 67.600512 | 19.647602 | 76.37s |
| Pythia-160M | 182.160394 | 292.206888 | 141.512514 | 40.647879 | 81.30s |

Against v075, Qwen gains `+3.279166` native points and all three MHA models
also improve.  Attention is unchanged; the gain is Linear-only.  The root and
v076 archive SHA256 is
`C87B61C8A4A9F869A43EFDEECF7734A0A810EA0E5621D51826EC5E56A31ED0E4`.

### C78 interaction checks (not enabled)

Two follow-up arms were measured from the v076 root: full-width JDRQ and
all-shape gram64 together produced Qwen native `372.598123` / panel
`260.050784`, slightly below v076; group-level GQA reciprocal scaling produced
`369.071722` / `258.272056` from v075; both are rejected.  The active root
therefore keeps all-shape gram64 with projection-only JDRQ.

## C80 full gram64 coverage (current root)

Starting from the v076 all-shape policy, the dynamic gram64 budget was widened
in four committed steps: ratio/max-blocks `0.16/16` (`877db7d`), `0.32/32`
(`07cf5f6`), `0.64/64` (`50782a8`), and finally `1.0/128` (`45179eb`).
Every step was positive on Qwen and on the three MHA guardrail models; API time
did not materially change because the exact solver already exits on unchanged
blocks.

| model | native total | panel total | Linear | Attention | API time |
|---|---:|---:|---:|---:|---:|
| Qwen2.5-0.5B | 386.903134 | 265.372589 | 315.942615 | 70.960519 | 208.70s |
| GPT-2 small | 164.221204 | 212.834117 | 142.914968 | 21.306236 | 78.60s |
| OPT-125M | 91.605403 | 144.328377 | 71.957801 | 19.647602 | 75.26s |
| Pythia-160M | 188.695479 | 297.879706 | 148.047600 | 40.647879 | 77.60s |

The full-coverage parent is archived as
`solutions/20260830_v080_c80-gram64-full-coverage_scoreNA_timeNA/` with
SHA256 `62EC3DB74933986886D01751E5307E58DDC8F4007E56D9A484C239F74AE69813`.

## C84 full gram64 coordinate sweep (v084 current root)

After v080 reached full activation Gram-64 coverage, the next experiment kept
the legal `ratio=1.0` / `max_blocks=128` selection unchanged and increased only
the deterministic coordinate-sweep count per 64-dimensional block.  This tests
whether the earlier coverage gains were limited by the discrete solver budget,
without changing the online activation state or using output supervision.

| sweep | Qwen native | Qwen panel | Qwen Linear | Qwen Attention | API time |
|---:|---:|---:|---:|---:|---:|
| 1 (v080) | 386.903134 | 265.372589 | 315.942615 | 70.960519 | 208.70s |
| 2 | 390.780409 | 266.815028 | 319.819890 | 70.960519 | 238.30s |
| 3 | 391.684115 | 267.151228 | 320.723597 | 70.960519 | 256.34s |
| 4 | 391.956412 | 267.252529 | 320.995893 | 70.960519 | 285.40s |
| 5 (v084) | **392.055970** | **267.289567** | **321.095451** | 70.960519 | **309.09s** |

The sweep=5 guardrail native totals were GPT-2 `167.049503`, OPT `92.848854`,
and Pythia `190.277968`; all three were positive versus sweep4.  Qwen remains
about `110.91s` below the official `420s` limit.  Marginal gains are now small,
so sweep6 is deferred in favor of a structural V/QK or output-level discrete
solver experiment.  The current root and immutable archive are byte-identical
with SHA256
`A8A4427DBA95723570FBDEBCDA1E4EDDBF152A3693CC851E30A87368A02CA284`:
`solutions/20260830_v084_c84-gram64-sweep5_scoreNA_timeNA/`.

## C85 second JDRQ pass and Q(W)-metric audit (2026-08-30)

Two targeted experiments were run from the v084 parent and committed before
evaluation. First, `ff8861f` added a second deterministic Gauss--Seidel pass
over the legal hierarchy residual blocks. It kept the online activation state
frozen and retained pass-1/pass-2 candidates for robust product selection. On
the Qwen primary full run it reached native `391.982508`, panel `267.262237`,
Linear `321.021989`, Attention `70.960519`, API `313.57s`; this is below v084
panel `267.289567` and costs more time, so `4e9861e` reverts the feature in the
active root while preserving the experiment in history.

The same audit also tested whether replacing the activation Gram with a
quantized-weight Gram could close the product gap. Although the direct
`Q(W)^T Q(W)` candidate improved the local Qwen/heterogeneous proxy, the
runtime provenance guard correctly identified the `R_A R_W` cross-residual in
the activation path. It is therefore non-compliant and remains rejected.
Static W-only Gram and dense/proxy blends passed the guard but all fell below
v084 on the Qwen panel. No C85 candidate is promoted.

At the C85 checkpoint the active root was again the v084 behavior; the latest commits are
`ff8861f` (experiment) and `4e9861e` (revert). Focused release/compliance
tests remain `48 passed, 1 deselected`.

## C86 attention block-H final-lattice (v086)

C86 adds a shared head-local Hadamard candidate after Smooth-QK, centering and
permutation selection. Q heads in the same GQA group and their KV head reuse a
single deterministic sign pattern, so the continuous QK dot product is
preserved. The candidate uses the final offset/refinement lattice in the A1
output scorer; only legal integer configuration and static CPU signs are
returned. Implementation commits are `2c1cf85`, `31b99d6` (GQA sign alignment)
and `90844fe` (final-lattice scorer).

Qwen primary result: native `392.064774`, panel `267.307909`, Linear
`321.095451`, Attention `70.969323`, API `313.58s`; panel delta vs v084 is
`+0.018342`. Heterogeneous guardrail results are GPT-2 `169.829549`, OPT
`92.579685`, and Pythia `190.239876`; OPT Attention regresses slightly while
GPT-2 gains strongly and Pythia is nearly flat. The candidate is promoted under
the Qwen-primary/soft-guardrail policy and archived at
`solutions/20260830_v086_c86-attn-block-final_scoreNA_timeNA/`.
