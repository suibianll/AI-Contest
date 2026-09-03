# Cross-model GPT probe

- model: `gpt2` (`GPT2LMHeadModel`)
- protocol: `cross-model-probe-v1`; base codec: `proxy-v2`
- layers/hidden/heads: `12/768/12/64` (q/kv)
- roles: `['q', 'k', 'v', 'o', 'ffn_in', 'proj']`
- panel: `0 Linear + 4 Attention`
- this is an architecture stress test, not an official score and not mixed into Qwen proxy trend audits

| Linear mean | Attention mean | Overall mean | API total (s) | Wall (s) |
|---:|---:|---:|---:|---:|
| 0.000000 | 0.440957 | 0.440957 | 5.206 | 5.368 |

## Error-source decomposition

- Linear interpretation: `unknown`
- Linear W-only/A-only/Both/interaction: `0.000000` / `0.000000` / `0.000000` / `0.000000`
- Attention interpretation: `paired_qk_coupling_likely`
- Attention Q-only/K-only/V-only/QK-only/Both: `-24.238424` / `-33.230875` / `0.024680` / `0.404943` / `0.440957`

### Static Linear role/family gain

| Group | cases | gain |
|---|---:|---:|

Static Linear q/k/v are projection roles; the Attention Q/K/V control arms above are a separate dynamic path.
Full per-role/layer/length results are in JSON `case_scores` and `decomposition`.
