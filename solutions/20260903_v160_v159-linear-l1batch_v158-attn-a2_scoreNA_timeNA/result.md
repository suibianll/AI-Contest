# v160 — v159 Linear (L1 batch) + v158 Attention (A2 scale-aware K center + A1 cleanup)

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
- Attention A1 equivalent cleanup (2026-09-03, no output change): the K
  centering result depends only on `(k, center_mode, center_value)` and the
  Hadamard rotation signs only on `(kv_num_heads, head_dim, seed)`.  Both are
  now computed once per mode/seed inside `_run_selection` and shared across
  the whole candidate sweep via `precomputed_centered_k` /
  `precomputed_block_signs` on `_attention_candidate_metrics`.  Verified
  bit-exact: Qwen default 120/120 zero delta and GPT-2 60/60 zero delta
  against the A2 baseline; attention calibration 60.7 s → 59.7 s.
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

- Archived `solution.py` SHA256: `33B1D061…` (L1 + A2 + A1).  Final
  authoritative integration audit: `artifacts/official_eval/v160-final-
  integration-default.json` (`source_sha256` `33b1d061…`).
- Single-file, self-contained; exposes only the six required APIs.

## Local proxy-v2 evidence (mechanism/time diagnostics only)

Final complete default-panel integration audit (168 Linear + 120 Attention
cases), same cache, same device:
`artifacts/official_eval/v160-final-integration-default.json` +
`logs/official_eval/v160-final-integration-default.md`.

| Metric | v160 |
|---|---:|
| Linear mean gain | 0.633526 |
| Attention mean gain | 0.742354 |
| Overall mean | 0.678871 |
| API total | 290.7 s |
| Wall time | 318.4 s |

| API | seconds | calls |
|---|---:|---:|
| `hif4_calibration_and_quantize_weight` | 166.64 | 168 |
| `hif4_dynamic_quantize_activation` | 60.65 | 168 |
| `hif4_calibration_attention` | 60.0 | 24 |
| `hif4_dynamic_quantize_q` | 1.43 | 120 |
| `hif4_dynamic_quantize_k` | 1.09 | 120 |
| `hif4_dynamic_quantize_v` | 0.87 | 120 |

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

- v160 is the first archived version that carries the L1 batched Linear
  encoding, the A2 attention change, and the A1 equivalent cleanup.  No
  further mechanism change is archived under v159.
- L2 bounded-complexity ablations (seeds/sizes/RMS-smooth/wide-alphas) were
  all REJECTED on the Qwen compact paired panel — every block-smooth search
  dimension is load-bearing; no ablation was merged.

## A1 evidence (equivalent cleanup, bit-exact)

- Qwen default pair `v160-a1-attn-default-candidate.json` vs the A2 build:
  120/120 zero delta, attention_mean unchanged 0.742354, calibration
  60.7 s → 59.7 s (24 layers), API 64.1 s → 63.1 s.
- GPT-2 pair `v160-a1-attn-gpt2-candidate.json` vs `v159-attn-gpt2-parent`:
  60/60 zero delta, attention_mean 0.389583.
- Six-API call counts unchanged (calibration_attention 24, dynamic q/k/v 120
  each); only calibration wall time decreased.
