# v032 / C40 robust Block-LDLQ

- Date: 2026-08-28
- Parent: v031 / C39-FW (`14613 / 159.2s` official)
- Source SHA256: `D24BC94F513907CBE97B43865973D1498133D8B9264FAF12661836FF65AAB656`
- Official score/time: pending
- Compliance: activation-only Hessians; no `A@W` or output-residual fitting

## Isolated algorithm change

At the same wide-layer scope and 25% budget as C39, adjacent 64-channel blocks
are coupled into 128-channel superblocks. Each conditional target is re-solved
with the full scale/hierarchy/GPTQ FULL64 solver and accepted only when pooled
and independent-fold Hessian objectives both improve.

## Local matrix

| Case | C39-FW | C40 | Delta |
|---|---:|---:|---:|
| offset 0 | 0.5357 | 0.5393 | +0.36pp |
| offset 97 | 0.5213 | 0.5248 | +0.35pp |
| offset 193 | 0.5385 | 0.5445 | +0.60pp |
| offset 389 | 0.5312 | 0.5368 | +0.56pp |
| amax4 | 0.4740 | 0.4754 | +0.14pp |
| pow2 | 0.5521 | 0.5550 | +0.29pp |

CUDA algorithm-stage was `45.32s` versus paired C39 `27.54s`; CPU
algorithm-stage was `100.05s`. Default causal Attention remained `0.4497`.
The full test suite passed: `66 passed`.

See [the detailed algorithm record](../../docs/C40-robust-block-ldlq.md).
