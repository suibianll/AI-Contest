# Cross-model GPT probe

- model: `gpt2` (`GPT2LMHeadModel`)
- protocol: `cross-model-probe-v1`; base codec: `proxy-v2`
- layers/hidden/heads: `12/768/12/64` (q/kv)
- roles: `['q', 'k', 'v', 'o', 'ffn_in', 'proj']`
- panel: `0 Linear + 60 Attention`
- this is an architecture stress test, not an official score and not mixed into Qwen proxy trend audits

| Linear mean | Attention mean | Overall mean | API total (s) | Wall (s) |
|---:|---:|---:|---:|---:|
| 0.000000 | 0.389583 | 0.389583 | 18.342 | 22.145 |

## Error-source decomposition

- Linear interpretation: `unknown`
- Linear W-only/A-only/Both/interaction: `0.000000` / `0.000000` / `0.000000` / `0.000000`
- Attention interpretation: `paired_qk_coupling_likely`
- Attention Q-only/K-only/V-only/QK-only/Both: `-27.367752` / `-29.002071` / `0.016191` / `0.367155` / `0.389583`

### Static Linear role/family gain

| Group | cases | gain |
|---|---:|---:|

Static Linear q/k/v are projection roles; the Attention Q/K/V control arms above are a separate dynamic path.
Full per-role/layer/length results are in JSON `case_scores` and `decomposition`.
