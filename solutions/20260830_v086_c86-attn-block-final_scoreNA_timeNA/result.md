# v086 / C86 attention block-smooth final-lattice candidate

- Date: 2026-08-30
- Parent: v084 / C84 full Gram-64 coverage + five coordinate sweeps
- Commit: `90844fe` (with GQA sign alignment in `31b99d6`)
- Source SHA256: `E7A16D6991DBB70A593FBE87D0C5D1D8FD38F801665354A01FFAF2F0A96F03CD`
- Official score/time: `NA` / `NA`

## Mechanism

C86 adds a shared head-local Hadamard candidate to attention Q/K calibration.
The same block size and deterministic sign pattern are used for each Q head
group and its corresponding KV head, so the continuous QK dot product is
unchanged. Candidates use block sizes 4, 8 and 16 (seed 0), are ranked with
the final offset/refinement lattice on the calibration output scorer, and only
the winning integer pair plus static signs enter Q/K state. No Linear
`activation_state`, A@W product or test output is used by this path.

## Local real-model evaluation

Configuration: `amax6 / seq128 / calib2 / test4 / cache_mode=read /
algorithm-device=cuda`; scores are local relative diagnostics, not official
score conversions.

| model | native total | panel total | Linear | Attention | API time |
|---|---:|---:|---:|---:|---:|
| Qwen2.5-0.5B | **392.064774** | **267.307909** | 321.095451 | **70.969323** | 313.58s |
| GPT-2 small | 169.829549 | 226.872764 | 145.743266 | **24.086283** | 121.01s |
| OPT-125M | 92.579685 | 144.286224 | 73.201252 | 19.378433 | 122.22s |
| Pythia-160M | 190.239876 | 299.094679 | 149.630088 | 40.609788 | 123.18s |

Relative to v084, Qwen panel improves by `+0.018342` and remains below the
420-second primary limit. GPT-2 Attention improves substantially; OPT has a
small Attention regression and Pythia is nearly unchanged, so the mechanism is
promoted as a Qwen-primary candidate with soft guardrail caveats.

## Verification

```text
python -m py_compile solution.py
python -m pytest -q tests/test_release_candidate.py tests/test_reference_hif4.py \
    tests/test_linear_compliance_guard.py tests/test_jdrq.py \
    -k "not local_holdout_offsets"       # 48 passed, 1 deselected
```
