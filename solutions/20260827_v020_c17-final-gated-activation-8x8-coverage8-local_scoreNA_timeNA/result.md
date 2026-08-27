# v020 — C17 Final Gated Activation 8×8 Coverage 8%

- Date: 2026-08-27
- Candidate ID: `C17`
- Parent: `C14`
- Unique mechanism: raise only the gated activation 8×8 residual coverage from 2% to 8%; retain the calibration gate, one sweep, cap 4096, dense-weight Gram and all other paths.
- Source SHA256: `C29E71C332E41E262B94FF68454CEB1F1589EE932FB4E1D55C5F221CFD060766`
- Parent SHA256: `EC246A8941ACBE4A6B1B085F44B9067F852456C4A0272C01266E1298D4CC6D45`
- Local status: `local-champion`
- Official status: `unavailable`

## Development result

Offset 0, amax6, CUDA:

| q | k | v | o | fc | proj | Linear mean delta |
|---:|---:|---:|---:|---:|---:|---:|
| +0.25pp | +0.31pp | +0.10pp | +0.46pp | +0.27pp | +0.32pp | +0.285pp |

- Candidate Linear mean `0.5890`, versus C14 `0.5861`.
- Attention remains exactly C14 (`0.4497/0.4942`).
- CUDA algorithm-stage `24.63s`, versus C14 `24.99s` on its development run.

## Fixed local matrix

| Case | C14 Linear mean | C17 Linear mean | Delta | Component safety |
|---|---:|---:|---:|---|
| amax6 offset 0 | 0.5861 | 0.5890 | +0.285pp | all positive |
| amax6 offset 97 | 0.5671 | 0.5696 | +0.257pp | all positive |
| amax6 offset 193 | 0.5860 | 0.5888 | +0.282pp | all positive |
| amax6 offset 389 | 0.5839 | 0.5867 | +0.280pp | all positive |
| amax4 offset 0 | 0.4900 | 0.4927 | +0.270pp | all positive |
| pow2 offset 0 | 0.5493 | 0.5524 | +0.302pp | all positive |

- All 36 recorded Linear component means improve over C14.
- GQA offset 193 Attention remains exactly C14 (`0.4169/0.4928`).
- Same-environment CPU pair: C17 `63.96s`, C14 `62.40s`, ratio `1.025`.
- Ten release tests pass; root and archive source hashes match exactly.

## Decision

`accepted as local Champion`. C17 clears the global gain, component safety, fixed-matrix and timing gates. Per preregistration, fixed activation 8×8 coverage tuning is now closed at 8%.

Next candidate: add a bounded block-local activation/weight-error cross term to the 8×8 coordinate objective, while retaining C17's calibration gate. This targets the part of final output MSE that a pure activation Gram cannot represent.
