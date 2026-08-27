# v014 — C11 Wide Activation 8×8 Residual

- Date: 2026-08-27
- Candidate ID: `C11`
- Parent: `C10`
- Unique mechanism: for activation states wider than 1024, store an 8×8 `W^T W` Gram and apply one `H·e` coordinate sweep to the highest-loss 2% of complete 8-channel groups, capped at 4096 groups.
- Source SHA256: `292023260BD386060509E65BA2688B9F06B2E0EB555C0C5DC9454027A66381E6`
- Parent SHA256: `DD8587257299626718A24EB89013447DA9105E8884F391104A6B350607399E44`
- Local status: `local-champion`
- Official status: `unavailable`

## Development result

Offset 0, amax6, CUDA:

| Component | C10 | C11 | Delta |
|---|---:|---:|---:|
| q | 0.6347 | 0.6347 | 0.00pp |
| k | 0.6955 | 0.6955 | 0.00pp |
| v | 0.5963 | 0.5963 | 0.00pp |
| o | 0.5397 | 0.5397 | 0.00pp |
| fc | 0.4957 | 0.4957 | 0.00pp |
| proj | 0.5246 | 0.5277 | +0.31pp |
| Linear mean | 0.5811 | 0.5816 | +0.052pp |

Attention remains exactly C10 (`0.4497/0.4942`). CUDA algorithm-stage was `22.32s` versus C10 `21.97s`, ratio `1.016`.

## Fixed local matrix

| Case | C10 Linear mean | C11 Linear mean | Delta | proj delta | Attention |
|---|---:|---:|---:|---:|---|
| amax6 offset 0 | 0.5811 | 0.5816 | +0.052pp | +0.31pp | unchanged |
| amax6 offset 97 | 0.5628 | 0.5628 | +0.002pp | +0.01pp | unchanged |
| amax6 offset 193 | 0.5777 | 0.5815 | +0.375pp | +2.25pp | unchanged |
| amax6 offset 389 | 0.5759 | 0.5796 | +0.370pp | +2.22pp | unchanged |
| amax4 offset 0 | 0.4825 | 0.4828 | +0.035pp | +0.21pp | unchanged |
| pow2 offset 0 | 0.5445 | 0.5451 | +0.068pp | +0.41pp | unchanged |

- All six configurations improve; q/k/v/o/fc are field-for-field unchanged at reported precision.
- GQA offset 193 Attention remains exactly C10 (`0.4169/0.4928`).
- Same-environment CPU pair: C11 `60.02s`, C10 `58.93s`, ratio `1.019`.
- Nine release tests pass. Wide 4×4 and 8×8 Grams use two legal CPU state tensors and 36,864 total elements.

## Decision

`accepted as local Champion`. The new residual produces repeatable proj gains in all fixed configurations, preserves non-target paths, and stays within the timing gate.

Next candidate: a strictly bounded 16×16 activation residual on top of C11, with lower 1% coverage and a separate target-specific gate. This continues the activation correlation ladder without reopening saturated weight-group tuning.
