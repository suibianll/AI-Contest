# v075 C76 GQA head-local rotation

## Snapshot

- Date: 2026-08-30
- Parent: v074 C75 rowwise JDRQ + wide gram64 hierarchy
- Root/archive SHA256: `DCA23116D76033A7EB5A04C5CC7EF003A52995905261699B2D06883D4C0BE4A4`
- Official score/time: `NA` (no official submission for this snapshot)
- Local run: CUDA, `amax6`, `seq128`, `calib2`, `test4`, cache read

## Active change

C76.4 adds a head-local signed Hadamard rotation search with block sizes
H16/H32/H64 and four deterministic sign seeds. Q heads in one GQA group and
their shared KV head receive the same orthogonal transform, so the continuous
Q·K score is unchanged. The candidate is selected only from the real deployed
Q/K quantizer on calibration prefixes. The first release enables it for GQA
(`q_num_heads != kv_num_heads`) and leaves MHA on the v074 path; this is a
structural width/head-count policy, not a model-name exception.

The C76.1 independent permutation, C76.2 output-Fisher importance and C76.3
reciprocal temperature arms remain implemented as disabled research switches
after their Qwen calibration-to-test regressions. They are not part of the
released score path.

## Local evaluator observations

| model | native total | panel proxy | Linear | Attention | API time |
|---|---:|---:|---:|---:|---:|
| Qwen2.5-0.5B (GQA rotation) | 369.344509 | 258.840363 | 298.383991 | 70.960519 | 188.06s |
| GPT-2 small (MHA, rotation skipped) | 158.550907* | 207.911984* | 137.244671* | 21.306236* | 68.96s* |
| OPT-125M (MHA, rotation skipped) | 85.736733* | 139.234046* | 66.089132* | 19.647602* | 69.03s* |
| Pythia-160M (MHA, rotation skipped) | 179.446007* | 289.850650* | 138.798128* | 40.647879* | 67.30s* |

`*` MHA values are the v074 rowwise2 measurements (the GQA-only gate makes
the code path identical), while Qwen is the direct v075 run. Native/panel
values are local relative diagnostics, not official score conversions.

## Verification

```text
python -m py_compile solution.py evaluator/jdrq_diagnostics.py tests/test_jdrq.py
python -m pytest -q tests/test_jdrq.py tests/test_linear_compliance_guard.py \
    tests/test_reference_hif4.py tests/test_release_candidate.py \
    -k "not local_holdout_offsets"       # 48 passed, 1 deselected
```
