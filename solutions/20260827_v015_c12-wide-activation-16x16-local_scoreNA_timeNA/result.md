# v015 — C12 Wide Activation 16×16 Residual

- Date: 2026-08-27
- Candidate ID: `C12`
- Parent: `C11`
- Unique mechanism: after C11's 4×4 and 8×8 activation solves, add one 16×16 `H·e` sweep to the highest-loss 1% of wide activation groups, capped at 2048 groups.
- Source SHA256: `C9A9F95EF7C77C8CBB55BDFF57C7E315AC9BCEC1444D17F1A1079793DE911118`
- Parent SHA256: `292023260BD386060509E65BA2688B9F06B2E0EB555C0C5DC9454027A66381E6`
- Local status: `local-accepted-not-promoted`
- Official status: `unavailable`

## Development result

Offset 0, amax6, CUDA:

| Component | C11 | C12 | Delta |
|---|---:|---:|---:|
| q | 0.6347 | 0.6347 | 0.00pp |
| k | 0.6955 | 0.6955 | 0.00pp |
| v | 0.5963 | 0.5963 | 0.00pp |
| o | 0.5397 | 0.5397 | 0.00pp |
| fc | 0.4957 | 0.4957 | 0.00pp |
| proj | 0.5277 | 0.5284 | +0.07pp |
| Linear mean | 0.5816 | 0.5817 | +0.012pp |

- Attention remains exactly C11 (`0.4497/0.4942`).
- CUDA algorithm-stage `22.80s`, versus C11 `22.32s`, ratio `1.022`.
- Nine release tests passed with three legal wide activation Gram state tensors.

## Decision

The residual is positive but below the preregistered `+0.2pp` proj gate. Fixed regression and CPU timing were skipped. C11 remains Champion and activation group-size expansion stops at 8×8.

Next candidate: retain C11's proven 8×8 residual but extend its eligibility from only 3072-wide inputs to all block-aligned Linear activations, testing whether q/k/v/o/fc can benefit without adding a larger correlation scale.
