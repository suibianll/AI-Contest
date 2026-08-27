# v018 — C15 Quantized-Weight Activation Gram

- Date: 2026-08-27
- Candidate ID: `C15`
- Parent: `C14`
- Unique mechanism: derive activation 4×4/8×8 quadratic Grams from the deployed quantized weight (`W_hat^T W_hat`) instead of `W_smooth^T W_smooth`; all gates and budgets remain unchanged.
- Source SHA256: `2E3B69776A042485C9408EE4A4493D6F4D1E41879154B8DEA9B714CE4C6B1BEB`
- Parent SHA256: `EC246A8941ACBE4A6B1B085F44B9067F852456C4A0272C01266E1298D4CC6D45`
- Local status: `local-accepted-not-promoted`
- Official status: `unavailable`

## Development result

Offset 0, amax6, CUDA:

| q | k | v | o | fc | proj | Linear mean delta |
|---:|---:|---:|---:|---:|---:|---:|
| -0.02pp | 0.00pp | +0.01pp | +0.05pp | -0.01pp | -0.03pp | ~0.000pp |

- Candidate scores: `0.6401/0.7067/0.5970/0.5462/0.4992/0.5274`.
- Attention remains exactly C14 (`0.4497/0.4942`).
- CUDA algorithm-stage `25.62s`, versus C14 `24.99s`.
- Ten release tests passed.

## Decision

The more literal operator Gram redistributes tiny component effects but produces no net Linear gain and is below the `+0.2pp` gate. Fixed regression and CPU timing were skipped. C14 remains Champion.

Next candidate: keep C14's dense-weight Gram and calibration safety gate, and raise only activation 8×8 coverage from 2% to 4% to measure remaining residual headroom.
