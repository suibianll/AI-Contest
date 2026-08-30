# Plans and process documents

Current compliance wording: offline Linear calibration may form `A@W` to
optimize an offline quantizer, especially `Q(W)`. It must not use that output
or residual to fit, select, or infer online `Q(A)`, and must not put it in
`activation_state`. This target-specific rule supersedes any older blanket
wording found in archived planning documents.

This directory contains implementation plans plus reusable process documents.
The clean Gram-hierarchy root was implemented after the JDRQ plan was written;
therefore the JDRQ file is now a research/rollback reference rather than a
literal description of the active root. The current measured behavior is kept
in [`docs/current-solution-status.md`](../../current-solution-status.md).

- `2026-08-26-solution-archive-workflow.md`: immutable candidate archive and
  result-recording workflow.
- `2026-08-29-hif4-jdrq-36000-implementation-plan.md`: **superseded research
  plan**. It specifies the JDRQ ceiling diagnostic and candidate routes that
  remain available for future experiments, but it is not a statement that the
  current root contains JDRQ output distillation.
- `2026-08-29-hif4-linear-22000-optimization-plan.md`: superseded historical
  algorithm plan. Do not use either plan to infer the current root behavior;
  use the current status report and `solution.py`.

Past optimization plans have been moved to `../archive/plans/`. They are
historical records, not instructions for future exploration. New optimization
work should start from the current `solution.py`, current measurements, the
official compliance rules, and the revised 420-second (7 minute) official
runtime limit.
