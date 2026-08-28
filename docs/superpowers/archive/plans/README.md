# Archived optimization plans

The documents in this directory are retained only as historical records.
They contain superseded hypotheses, thresholds, candidate ordering, runtime
budgets, and stopping rules. In particular, the C21-C follow-up plan produced
false-negative decisions by treating provisional gates as hard limits and
must not be used to direct future optimization.

Only the following constraints remain authoritative:

1. Do not compute `A@W`, directly or indirectly, and use it to fit or infer
   `Q(A)`.
2. Keep the HiF4 output, API, state, and numerical behavior legal.
3. Keep the final official evaluation strictly below 300 seconds.
4. Do not tune from holdout or official-score feedback.

All other thresholds and claims in these archived files are historical.
Current algorithm status is documented in the repository root README and the
latest execution log.
