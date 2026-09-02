# Local evaluation artifact contract

All JSON files in this directory are local evidence. None is an official score or an official
runtime conversion. Before comparing two files, read their `protocol` and `evaluation_scope`.

| scope | use | may rank candidates? |
|---|---|---:|
| `default-panel` | Same `proxy-v2` cache, deterministic 168 Linear + 120 Attention panel | yes, only within the identical cache/panel/device |
| `compact-generalization-panel` | Four depth-spread layers, paired validation/test holdouts, selected calibration states, robust Linear tails | no |
| `effect-panel` | Paired mechanism diagnosis (56 Linear + 5 Attention) | no |
| `paired-json-replay` | Zero-API replay of two identical effect panels | no |
| `full-stress` | Full Cartesian stress/regression check | no |
| `smoke-prefix` | Interface, legality and local sanity check | no |
| `research-oracle` | Bounded teacher/oracle or feature probe; no candidate API | no |
| `official-shape-v1` | Immutable historical evidence | no |
| GPT-2 / external hif4 | Cross-model or external-implementation probe | no |

Scenario-isolated runs prefix the scope with `linear-only-` or `attention-only-`. Use
`--linear-only` for Linear mechanisms and `--attention-only` for Attention mechanisms. These
runs execute the complete shared calibration graph for the selected side and call **zero APIs**
from the other side; paired comparisons require a parent JSON produced with the same scenario.
Run both sides only for an explicit end-to-end integration audit.

The evaluator writes `evaluation_scope.kind`, `intent`, `comparable_for_proxy_ranking` and
`official_score_equivalent` into new results. `official_score_equivalent` is always `false`.
Research workbench JSON may instead use a top-level `diagnostic`/`scope` field because it does not
call the six-API evaluator; treat it as `research-oracle` and never rank it with evaluator JSON.
The only authoritative official facts are the user-confirmed records in
`docs/current-solution-status.md` and `solutions/README.md`.

Legacy `official-shape-v1` JSON (36 files, including `archive-official-shape-v1.json`) has been
moved to the `legacy-v1/` subdirectory for physical isolation. It is immutable historical evidence:
do not glob it into proxy-v2 analysis, do not rank with it, and do not update it with new runs.

Fast iteration rule:

1. Run interface/legality smoke once for a new module.
2. Select exactly one scenario and run a paired `compact-generalization-panel` against a
   same-scenario immutable parent JSON. Linear compact runs use 28 Weight states and 56 dynamic
   cases: four depth-spread layers × seven roles × paired validation/test holdouts.
3. Inspect focus role, controls, median/q25/worst-quartile, negative cases, cross-holdout sign
   consistency, W/A or Q/K/V arms, worst layer and API delta. Do not promote on mean alone.
4. Run the selected side's `default-panel` only after the mechanism is explainable; use the old
   `effect-panel` only when a full calibration graph with reduced dynamic scoring is specifically
   needed, and reserve `full-stress` for release
   candidates or explicit regression checks.

Teacher/oracle workbench JSON is research evidence, not a candidate score. Its wall time must not
be compared with the six API timing fields.
