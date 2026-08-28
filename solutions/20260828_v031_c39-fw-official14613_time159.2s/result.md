# v031 / C39-FW official result

- Date: 2026-08-28
- Parent: v025 / C21-C (`14437 / 166.6s`)
- Official score: **14613**
- Official runtime: **159.2s**
- Official delta: **+176 points / -7.4s**
- Source SHA256: `B8C9F2A4EB6553367DD17E73D30836AC8911DBEF33759FA8CF95E8C629317A71`
- Compliance: activation-only statistics; no `A@W` computation or output-residual fitting

## Algorithmic change

C39-FW retains the proven C21-C activation path and enables FULL64 Hessian-aware
weight refinement only on wide FFN `fc`/`proj` layers. The attention projections
remain unchanged, isolating the effect of the wide-layer weight solver.

## Local evidence

| Linear case | C21-C | C39-FW | Delta |
|---|---:|---:|---:|
| offset 0 | 0.5311 | 0.5357 | +0.46pp |
| offset 97 | 0.5148 | 0.5213 | +0.65pp |
| offset 193 | 0.5319 | 0.5385 | +0.66pp |
| offset 389 | 0.5235 | 0.5312 | +0.77pp |
| amax4 | 0.4663 | 0.4740 | +0.77pp |
| pow2 | 0.5454 | 0.5521 | +0.67pp |

Attention remained `0.4497`; CUDA algorithm-stage runtime was `27.47s`.

## Decision

C39-FW is the current compliant official champion and the immutable parent for
the next algorithm experiment. C38 is not a valid parent: its aggressive
activation and narrow-layer FULL64 combination scored only `14092 / 170.57s`.

The next candidate must improve the solver itself. The selected direction is a
robust Block-LDLQ conditional re-solve across adjacent 64-dimensional blocks,
guided only by activation Hessians and accepted against independent activation
folds. Coverage or gate changes alone do not qualify as a new candidate.
