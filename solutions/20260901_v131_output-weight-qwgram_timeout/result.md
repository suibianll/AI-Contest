# v131 official update

Status: **TIMEOUT / OFFICIAL FAILURE (user-confirmed)**.

- Parent: v130
- Change: use the final deployed `Q(W)^T Q(W)` block Gram for online Linear
  activation refinement.
- Local `official-shape-v1`: Linear `0.473131`, Attention `0.836579`, API
  `294.835s`, wall `317.708s`.
- Official outcome (user confirmed 2026-09-01): **timeout**, time `>300s`, no
  score returned.

Like v129/v130, v131 still uses the expensive Attention calibration and dynamic
Q/K Gram path.  Its timeout is therefore additional evidence for the v138
Attention rollback, not evidence that the `Q(W)` Linear Gram itself is invalid.
