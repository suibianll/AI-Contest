# Archived optimization plans

> **Rule correction (2026-08-28):** Some documents below were written under
> an over-broad interpretation that all calibration-time `A@W` was forbidden.
> That interpretation is historical and superseded. The official boundary is
> target-specific: offline calibration may use `A@W` to optimize an offline
> quantizer, especially `Q(W)`, but `A@W`, its output, or its residual may not
> be used to fit, select, or infer online `Q(A)`, and may not enter
> `activation_state`. Read the current root README and evaluator guard as the
> authoritative wording.

The documents in this directory are retained only as historical records.
They contain superseded hypotheses, thresholds, candidate ordering, runtime
budgets, and stopping rules. In particular, the C21-C follow-up plan produced
false-negative decisions by treating provisional gates as hard limits and
must not be used to direct future optimization.

Only the following constraints remain authoritative:

1. Offline calibration may compute `A@W` for an offline quantizer objective,
   but must not use it to fit, select, or infer `Q(A)` or populate
   `activation_state`.
2. Keep the HiF4 output, API, state, and numerical behavior legal.
3. Keep the final official evaluation strictly below 300 seconds.
4. Do not tune from holdout or official-score feedback.

All other thresholds and claims in these archived files are historical.
Current algorithm status is documented in the repository root README and the
latest execution log.
