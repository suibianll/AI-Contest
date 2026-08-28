# Historical specifications

Some specifications predate the 2026-08-28 rule clarification. Their blanket
statements that calibration must never form `A@W` are historical. The current
official boundary allows offline `A@W` objectives for an offline quantizer,
especially `Q(W)`, but never allows the output or residual to fit, select, or
infer online `Q(A)` or to enter `activation_state`.

The files in this directory describe earlier architecture and workflow design
decisions. They are retained for provenance and are not active optimization
instructions. Current work starts from root `solution.py`, the root README,
and reproducible measurements. Superseded execution plans are under
`../archive/plans/`.
