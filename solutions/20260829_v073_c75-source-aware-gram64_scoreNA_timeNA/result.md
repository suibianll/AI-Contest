# C75 source-aware activation proposals + gram64 activation refinement

- Parent: v072 / C74 fixed-Q(A) hierarchy residual.
- Unique mechanisms:
  - add legal E6M2 scale proposals derived from the four NVFP4 E4M3 source
    scales in each 64-channel group;
  - build CPU-only, per-64-group static `W.T @ W` slices during calibration
    and use one exact signed-mantissa lattice sweep for selected activation
    hard blocks;
  - enable the full-64 activation state only for the shape-derived
    down-projection path (`out_features < in_features`), while keeping the
    parent path for square/up-projection layers;
  - rerun fixed-state JDRQ after the activation state is finalized.
- Compliance: `activation_state` contains only static quantization state and
  the bounded gram64 tensor; products, residuals and candidate scores remain
  calibration-local. No online output is used to infer `Q(A)`.
- Root/archive SHA256: `A0DCE5D79DA931D5B67FACCBA47226B6C8FCE9FC9551200ED86A3693A1E464DA`

## Local screen

The project-only gram64 rerun is complete for GPT-2; Qwen/OPT/Pythia are
being rerun after the shape-derived gate. Earlier all-shape gram64 numbers
are retained as directional evidence, not as the v073 final score.

| model | configuration | native total | API time | status |
|---|---|---:|---:|---|
| GPT-2 small | source-aware + project-only gram64 | 158.561896 | 59.64s CUDA | measured; better than all-shape gram64 and parent source arm |
| Qwen2.5-0.5B | source-aware + project-only gram64 | NA | NA | rerun pending |
| OPT-125M | source-aware + project-only gram64 | NA | NA | rerun pending |
| Pythia-160M | source-aware + project-only gram64 | NA | NA | rerun pending |

Directional all-shape gram64 measurements were GPT-2 `159.690510`, Qwen
`363.937585`, OPT `87.315586`, and Pythia `182.048492`; the GPT-2 attribution
run showed that square q/k/v/o paths caused migration noise, motivating the
shape-derived project-only policy.

## Verification

```text
python -m py_compile solution.py evaluator/jdrq_diagnostics.py tests/test_jdrq.py
python -m pytest -q tests/test_jdrq.py tests/test_release_candidate.py \
    -k "not local_holdout_offsets"   # 25 passed, 1 deselected
```

The archive is a reproducible candidate snapshot; it is not an official
score claim until the remaining project-only model reruns finish.
