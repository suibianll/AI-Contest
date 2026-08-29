# Active plans

Current compliance wording: offline Linear calibration may form `A@W` to
optimize an offline quantizer, especially `Q(W)`. It must not use that output
or residual to fit, select, or infer online `Q(A)`, and must not put it in
`activation_state`. This target-specific rule supersedes any older blanket
wording found in archived planning documents.

This directory contains the current implementation plan plus reusable process
documents. The JDRQ plan below supersedes the earlier 22000-point algorithm
plan; the older file remains in place only so prior candidate decisions can be
traced accurately.

- `2026-08-26-solution-archive-workflow.md`: immutable candidate archive and
  result-recording workflow.
- `2026-08-29-hif4-jdrq-36000-implementation-plan.md`: **authoritative active
  algorithm plan**. It specifies the JDRQ ceiling diagnostic, structured output
  distillation, complete HiF4 residual solver, activation upgrades, Attention
  search, tests, candidate archives, and anti-stagnation rules.
- `2026-08-29-hif4-linear-22000-optimization-plan.md`: superseded historical
  algorithm plan. Do not use it to choose new candidates.

Past optimization plans have been moved to `../archive/plans/`. They are
historical records, not instructions for future exploration. New optimization
work should start from the current `solution.py`, current measurements, the
official compliance rules, and the revised 420-second (7 minute) official
runtime limit.
