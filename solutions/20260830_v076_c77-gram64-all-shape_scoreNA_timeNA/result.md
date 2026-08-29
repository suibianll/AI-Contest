# v076 / C77 all-shape gram64 activation refinement

## Mechanism

`_ACTIVATION_GRAM64_PROJ_ONLY` is disabled.  The static transformed-weight
`W.T @ W` 64-block slices are therefore available for every Linear shape that
passes the existing width/state caps.  The tensor is calibration-only CPU
state; the dynamic path still returns the legal HiF4 five fields.  C76.4
GQA-only head-local H16/H32/H64 rotation remains active for Attention.

## Local paired evaluation

Configuration: `amax6 / seq128 / calib2 / test4 / cache_mode=read`, CUDA for
device and algorithm device.  Scores are local evaluator values and are not an
official-score conversion.

| model | native total | panel total | Linear | Attention | API time |
|---|---:|---:|---:|---:|---:|
| Qwen2.5-0.5B | 372.623675 | 260.060290 | 301.663157 | 70.960519 | 207.72s |
| GPT-2 small | 159.774232 | 208.973897 | 138.467995 | 21.306236 | 78.68s |
| OPT-125M | 87.248114 | 140.546008 | 67.600512 | 19.647602 | 76.37s |
| Pythia-160M | 182.160394 | 292.206888 | 141.512514 | 40.647879 | 81.30s |

Compared with v075, Qwen improves by `+3.279166` native points and every
heterogeneous model is also positive.  Attention values are unchanged; the
gain is entirely Linear activation refinement.  A combined experiment that
also opened full-width JDRQ reached Qwen panel `260.050784`, slightly below
this candidate, so JDRQ remains projection-only.

## Verification

Focused release/compliance/JDRQ/reference tests: `48 passed, 1 deselected`.

Root/archive SHA256:

`C87B61C8A4A9F869A43EFDEECF7734A0A810EA0E5621D51826EC5E56A31ED0E4`
