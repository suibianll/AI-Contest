# v139 official result correction

The user reported the official evaluation result for v139 as **15716 points / 202 seconds**.
This is a pass under the official end-to-end limit of `<300s`, and is one point above the
reported v138 result (`15715 / 208s`).

The local `official-shape-v1` proxy for v139 remains `linear_mean=0.5072782560`,
`attention_mean=0.7159419612`, API `193.3892126s`, wall `217.1957354s`. Those local values
are retained for reproducibility and are not substituted for the official score.

Decision: mark v139 **RETAINED / OFFICIAL PASS**. It is an official-result archive but is not
promoted to the active local root; v140 remains the active root pending its own official result.
