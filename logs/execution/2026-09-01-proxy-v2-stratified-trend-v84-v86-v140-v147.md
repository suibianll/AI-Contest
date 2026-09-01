# proxy-v2 stratified panel trend audit: v84 / v86 / v140 / v147

日期：2026-09-01

## Protocol

- evaluator: `evaluator/official_eval.py`, `proxy-v2`
- cache: `artifacts/official_eval/cache/qwen2.5-0.5b-proxy-v2.pt`
- model/data: the same Qwen2.5-0.5B and pinned validation/test holdout cache
- default stratified panel: 168 Linear + 120 Attention cases
- Linear: every 24 layer × 7 role exactly once; role/window assignment rotates over windows 0–4
- Attention: every layer over windows 0–4, covering lengths 10/128/512/1024/1024
- calibration: 168 Weight states + 24 Attention states, shared across dynamic cases
- error-source decomposition: enabled; it does not change score or candidate API calls

## Results

| Candidate | Official score | Official time | Local Linear | Local Attention | Local overall | API total (s) | Wall (s) |
|---|---:|---:|---:|---:|---:|---:|---:|
| v084 | 16517 | 252.563 | 0.448180 | 0.709391 | 0.557018 | 285.506 | 316.151 |
| v086 | 16744 | 222.700 | 0.448180 | 0.724735 | 0.563411 | 293.625 | 322.895 |
| v140 | 15838 | 207.000 | 0.570882 | 0.722287 | 0.633968 | 203.742 | 233.570 |
| v147 | 16579 | 211.000 | 0.570495 | 0.724735 | 0.634761 | 289.410 | 321.395 |

## Trend result

The pairwise audit over the four same-cohort official anchors is:

- 3 concordant pairs
- 3 inverted pairs
- status: `inversion_detected`

Concordant: v84→v86, v84→v147, v140→v147. Inverted: v84→v140, v86→v140,
v86→v147.

The useful aligned trend is v84→v86: Linear is unchanged and the local Attention mean rises
by about `0.0153`, matching the official increase from `16517` to `16744`. The main failures are
Linear changes: v140 raises local Linear by about `0.1227` over v86 and v147 raises it by about
`0.1223`, while the official scores fall by `906` and `165` respectively. Therefore the local
panel is not an official ranking proxy for the Linear family; its output is diagnostic only.

## Error-source reading

- v084/v086 Linear is identical (`0.448180`), as expected from their shared Linear path.
- v140/v147 Linear has a large positive paired output gain, but the isolated controls are invalid as
  independent deployable improvements: v140 has W-only gain `-210.30`, A-only gain `-123.79`, and
  Both gain `0.57088`, with interaction gain `334.66`. This is a strong
  `paired_coordinate_coupling_likely` signal, not evidence that either side alone is better.
- Attention is Q/K-coupled: for v86, Q-only/K-only gains are `-17.32/-34.66`, QK-only is
  `0.71863`, and V-only is approximately neutral (`-0.00066`). The evaluator labels this
  `paired_qk_coupling_likely`; optimization should preserve the Q/K pair and focus on the
  logits/softmax path rather than V.

## Decision

Keep this stratified panel for iteration speed and coverage. Do not promote a Linear algorithm from
`overall_mean` alone: v140/v147 prove that the remaining local ranking inversion is concentrated in
the Linear coupled-transform family. Use `decomposition.linear.by_role/by_layer/by_shape` to choose
the next controlled Linear experiment, with v86 Attention frozen. Keep `--full-cases` as an occasional
stress check, not as the normal iteration loop.
