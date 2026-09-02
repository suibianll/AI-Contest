# Evaluator scope cleanup — 2026-09-02

## Why the local view became misleading

The directory mixed four different purposes under similar-looking `linear_mean` and
`attention_mean` fields: historical `official-shape-v1`, current `proxy-v2`, prefix smoke runs and
paired effect panels. Full-stress runs and GPT-2/upstream hif4 probes added more numbers, while old
reports had no machine-readable scope. Reading aggregate means across those files made a diagnostic
panel look like a ranking panel and made local time look like the official 300-second gate.

## Change

`evaluator/official_eval.py` now writes an `evaluation_scope` object to every newly evaluated result,
and archive/replay outputs write the same contract:

- `default-panel`: only local proxy-ranking scope, and only for the identical cache/panel/device;
- `effect-panel` / `paired-json-replay`: paired mechanism diagnosis;
- `full-stress`: stress/regression only;
- `smoke-prefix`: interface/legality only;
- old v1, GPT-2 and external hif4: immutable history or cross-structure probes.

Every scope has `official_score_equivalent=false`. Reports print the scope and intent; archive JSON
contains a `scope_contract` string. Existing raw JSON is not rewritten.

## Verification

`py_compile` passed. A zero-API replay of the existing L3 parent and L2 probe produced
`artifacts/official_eval/l2-pair-probe-replay.json` with scope `paired-json-replay` and preserved
the original `0/16/40` Linear effect counts. No candidate API or cache was rerun for this check.
