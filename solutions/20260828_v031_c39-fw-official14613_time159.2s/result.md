# v031 / C39-FW official result

> 2026-08-29 更新：官方评测集改为 250 个 Linear case + 200 个 Attention
> case。下列新版结果覆盖旧评测集的 `14613 / 159.2s`；归档目录名保留，
> 以避免破坏既有历史链接。

- Date: 2026-08-28
- Parent: v025 / C21-C (`14437 / 166.6s`)
- Legacy official score/runtime (old panel): `14613 / 159.2s`
- Revised official score/runtime (250 + 200 cases): **`21864 / 161.3s`**
- Revised panel time limit: **`420s` (7 minutes)**
- Revised official delta versus v025/C21-C: not available from the supplied result
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

C39-FW was the compliant official champion under the legacy panel and remains
an immutable historical anchor. C38 is not a valid parent: its aggressive
activation and narrow-layer FULL64 combination scored only `14092 / 170.57s`.

Under the revised panel, C39-FW is a compliant official anchor rather than the
current local champion (v051/C47b is `22451 / 234s`). The next candidate must
improve the solver itself. The selected direction is a robust Block-LDLQ
conditional re-solve across adjacent 64-dimensional blocks, guided only by
activation Hessians and accepted against independent activation folds. Coverage
or gate changes alone do not qualify as a new candidate.
