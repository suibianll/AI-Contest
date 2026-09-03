# v160 — v159 Linear (L1 batch) + v158 Attention (A2 scale-aware K center)

## Scope

- Base Linear: v159 (`solutions/20260902_v159_linear-gptq17816_v158-attention_score17532_timeNA`),
  i.e. the user-provided 17816 Linear implementation with its GPTQ/AdaRound
  dependencies, plus the L1 bit-exact batch candidate-metrics encoding
  (commit ec18a88).
- Attention: retained from v158 and extended by A2 — the scale-aware
  quantization-aware K center (mode 4) is re-enabled for GQA
  (`_ATTN_SCALE_AWARE_CENTER_GQA = True`). Mode 4 was already in the runtime
  candidate set; the single switch lets it compete under GQA, gated by
  `_candidate_is_safe` so a degraded center can never regress the parent
  behaviour.
- No ROAB / L3 family / unconstrained permutation-scale search is included.
  Online path is unchanged: state saves a fixed center; dynamic Q/K/V stay a
  single center + encode.

## Official status

`score NA / time NA / runtime status unknown` — v160 has NOT been submitted to
the official evaluator.  The archived source is for local mechanism and
time-budget control only.  The v159 Linear component itself previously
received an official `17532 / timeNA` reply (pre-L1 source, SHA
`0508045A…`); the L1 re-encoding is bit-exact so the official score basis is
unchanged, but no official time was ever returned for it.

## Source identity

- Archived `solution.py` SHA256: `29AA1863…` (full SHA in
  `artifacts/official_eval/v160-integration-default.json` `source_sha256`).
- Single-file, self-contained; exposes only the six required APIs.

## Local proxy-v2 evidence (mechanism/time diagnostics only)

Complete default-panel integration audit (168 Linear + 120 Attention cases),
same cache, same device:
`artifacts/official_eval/v160-integration-default.json` +
`logs/official_eval/v160-integration-default.md`.

| Metric | v160 |
|---|---:|
| Linear mean gain | 0.633526 |
| Attention mean gain | 0.742354 |
| Overall mean | 0.678871 |
| API total | 296.0 s |
| Wall time | 324.5 s |

| API | seconds | calls |
|---|---:|---:|
| `hif4_calibration_and_quantize_weight` | 170.04 | 168 |
| `hif4_dynamic_quantize_activation` | 61.1 | 168 |
| `hif4_calibration_attention` | 61.43 | 24 |
| `hif4_dynamic_quantize_q` | 1.45 | 120 |
| `hif4_dynamic_quantize_k` | 1.06 | 120 |
| `hif4_dynamic_quantize_v` | 0.92 | 120 |

Per-side baselines and pairing evidence:

- Linear: `v159-l1-batched-default-parent.json` — same linear_mean 0.633526,
  bit-exact batch encoding; API 231.4 s vs 269.4 s pre-L1 (-14 %).
- Attention A2 default pair: `v160-a2-attn-default-candidate.json` vs
  `v159-attn-default-parent.json` — mean Δ gain +0.0066 (120 cases,
  17 positive / 3 negative / 100 zero); validation +0.0057, test +0.0080;
  all four test lengths positive. Known tails: layer 11 len-10 -0.170,
  layer 14 len-128/512 ≈ -0.02.
- GPT-2 cross-model gate: `v160-a2-attn-gpt2-candidate.json` — 60/60 zero
  delta (MHA is not affected by the GQA gate; no cross-model regression).
- Attention A2 calibration cost: 60.7 s vs 58.5 s parent (+2.2 s / 24 layers
  for the per-layer scale-aware center fixed-point solve).

## Version bookkeeping

- v160 is the first archived version that carries BOTH the L1 batched Linear
  encoding and the A2 attention change.  No further mechanism change is
  archived under v159.
- L2 bounded-complexity ablations (seeds/sizes/RMS-smooth/wide-alphas) were
  all REJECTED on the Qwen compact paired panel — every block-smooth search
  dimension is load-bearing; no ablation was merged.
