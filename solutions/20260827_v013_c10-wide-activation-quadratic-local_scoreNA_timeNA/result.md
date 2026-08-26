# v013 — C10 Wide Activation Quadratic

- Date: 2026-08-27
- Candidate ID: `C10`
- Parent: `C5`
- Unique mechanism: raise `_ACTIVATION_QUADRATIC_MAX_FEATURES` from 1024 to 4096 so 3072-wide FFN down-projection activations use the existing 4×4 `W^T W` Gram objective.
- Source SHA256: `DD8587257299626718A24EB89013447DA9105E8884F391104A6B350607399E44`
- Parent SHA256: `A093940D46BE4B3C3CA88B30CD4456DD112CAD1C5DE632FCDB0207A12D197288`
- Local status: `local-champion`
- Official status: `unavailable`

## Development result

Offset 0, amax6, CUDA:

| Component | C5 | C10 | Delta |
|---|---:|---:|---:|
| q | 0.6347 | 0.6347 | 0.00pp |
| k | 0.6955 | 0.6955 | 0.00pp |
| v | 0.5963 | 0.5963 | 0.00pp |
| o | 0.5397 | 0.5397 | 0.00pp |
| fc | 0.4957 | 0.4957 | 0.00pp |
| proj | 0.5192 | 0.5246 | +0.54pp |
| Linear mean | 0.5802 | 0.5811 | +0.090pp |

Attention remains exactly C5 (`0.4497/0.4942`). CUDA algorithm-stage was `21.97s` versus C5's recorded development run `20.81s`, ratio `1.056`.

## Fixed local matrix

| Case | C5 Linear mean | C10 Linear mean | Delta | Attention |
|---|---:|---:|---:|---|
| amax6 offset 0 | 0.5802 | 0.5811 | +0.090pp | unchanged |
| amax6 offset 97 | 0.5617 | 0.5628 | +0.107pp | unchanged |
| amax6 offset 193 | 0.5766 | 0.5777 | +0.112pp | unchanged |
| amax6 offset 389 | 0.5743 | 0.5759 | +0.163pp | unchanged |
| amax4 offset 0 | 0.4818 | 0.4825 | +0.065pp | unchanged |
| pow2 offset 0 | 0.5436 | 0.5445 | +0.085pp | unchanged |

- All six configurations improve; the change is isolated to the intended `proj` path.
- GQA offset 193 Attention remains exactly C5 (`0.4169/0.4928`).
- Same-environment CPU pair: C10 `50.99s`, C5 `51.25s`, ratio `0.995`; this is treated as timing parity, not a speed claim.
- Nine release tests pass. The 3072-wide Gram is one legal CPU state tensor with 12,288 elements.

## Decision

`accepted as local Champion`. C10 clears its target-specific proj gate, improves every fixed configuration, preserves all unaffected paths and stays within the 1.15 timing limit.

Next candidate: retain C10 and test a bounded 8×8 activation residual refinement only for wide activations. This follows the positive correlation signal without changing the weight or Attention mechanisms.
