# v080 / C80 full gram64 coverage

## Mechanism

The active all-shape gram64 refinement now uses `_ACTIVATION_GRAM64_MAX_RATIO =
1.0` and `_ACTIVATION_GRAM64_MAX_BLOCKS = 128`.  Every eligible 64-channel
activation block is scored with the static transformed-weight `W.T @ W` metric;
the legal hierarchy/mantissa update is accepted blockwise.  This remains
offline calibration data and a CPU state tensor only.  C76.4 GQA-only
head-local rotation and projection-only JDRQ are unchanged.

## Local paired evaluation

Configuration: `amax6 / seq128 / calib2 / test4 / cache_mode=read`, CUDA for
device and algorithm device.  Scores are local evaluator values, not official
score conversions.

| model | native total | panel total | Linear | Attention | API time |
|---|---:|---:|---:|---:|---:|
| Qwen2.5-0.5B | 386.903134 | 265.372589 | 315.942615 | 70.960519 | 208.70s |
| GPT-2 small | 164.221204 | 212.834117 | 142.914968 | 21.306236 | 78.60s |
| OPT-125M | 91.605403 | 144.328377 | 71.957801 | 19.647602 | 75.26s |
| Pythia-160M | 188.695479 | 297.879706 | 148.047600 | 40.647879 | 77.60s |

Relative to v076, Qwen improves by `+5.558080` native points and all three
MHA models also improve.  Relative to the immediately preceding 64-block
candidate, Qwen gains another `+1.229366` panel points.  Attention is unchanged
throughout; the gain is Linear activation refinement.  The intermediate
coverage changes are committed as `877db7d` (16), `07cf5f6` (32), and
`50782a8` (64).

## Verification

Focused release/compliance/JDRQ/reference tests: `48 passed, 1 deselected`.

Root/archive SHA256:

`62EC3DB74933986886D01751E5307E58DDC8F4007E56D9A484C239F74AE69813`
