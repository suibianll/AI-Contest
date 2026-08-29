# v074 C75 rowwise JDRQ + wide gram64 hierarchy

## Snapshot

- Date: 2026-08-30
- Parent: v073 source-aware activation + project-only gram64
- Root/archive SHA256: `7789A0487915EE1860EECA2736311BDD1E357BF86E5528805472182F51B944CC`
- Official score/time: `NA` (no official submission for this snapshot)
- Local run: CUDA, `amax6`, `seq128`, `calib2`, `test4`, cache read

## Active changes

1. Rowwise JDRQ hierarchy: each output row selects its own high-leverage 64-block
   candidates, with a two-block budget for widths up to 4096 and a soft held-out
   calibration-window ranking term.
2. Wide activation gram64: down-projections (`out < in`) up to 8192 channels use
   static 64x64 slices of `W.T@W`, followed by a legal E6M2/lv2/lv3 hierarchy
   beam and one signed-mantissa sweep. Only CPU gram64 state is returned.
3. H32/H64 are optional operand-local candidates. The experimental output-product
   reranker is disabled in the release path because the runtime provenance audit
   flags its intermediates as residual cross contractions; JDRQ's offline static
   `A@W` objective remains enabled.
4. NVFP4 source-scale proposals remain proposal-only for online `Q(A)`; every
   activation-state change rebuilds the frozen-Q(A) JDRQ static `Q(W)` candidate.

## Local evaluator observations

| model | native total | panel proxy | Linear | Attention | API time |
|---|---:|---:|---:|---:|---:|
| GPT-2 small | 158.550907 | 207.911984 | 137.244671 | 21.306236 | 68.96s |
| OPT-125M | 85.736733 | 139.234046 | 66.089132 | 19.647602 | 69.03s |
| Pythia-160M | 179.446007 | 289.850650 | 138.798128 | 40.647879 | 67.30s |
| Qwen2.5-0.5B | 361.503707 | 242.505358 | 298.383991 | 63.119717 | 179.27s |

The native/panel values are local relative diagnostics and must not be
interpreted as official score conversions. All API times are below 420 seconds.

## Verification

```text
python -m py_compile solution.py evaluator/jdrq_diagnostics.py tests/test_jdrq.py
python -m pytest -q tests/test_jdrq.py tests/test_linear_compliance_guard.py \
    tests/test_reference_hif4.py tests/test_release_candidate.py \
    -k "not local_holdout_offsets"       # 48 passed, 1 deselected
```
