# Active plans

Current compliance wording: offline Linear calibration may form `A@W` to
optimize an offline quantizer, especially `Q(W)`. It must not use that output
or residual to fit, select, or infer online `Q(A)`, and must not put it in
`activation_state`. This target-specific rule supersedes any older blanket
wording found in archived planning documents.

This directory contains only reusable process documents that remain active.

- `2026-08-26-solution-archive-workflow.md`: immutable candidate archive and
  result-recording workflow.

Past optimization plans have been moved to `../archive/plans/`. They are
historical records, not instructions for future exploration. New optimization
work should start from the current `solution.py`, current measurements, the
official compliance rules, and the 300-second official runtime limit.
