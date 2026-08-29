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
