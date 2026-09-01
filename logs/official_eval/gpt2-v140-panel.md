# Cross-model GPT probe

- model: `gpt2` (`GPT2LMHeadModel`)
- protocol: `cross-model-probe-v1`; base codec: `proxy-v2`
- layers/hidden/heads: `12/768/12/64` (q/kv)
- roles: `['q', 'k', 'v', 'o', 'ffn_in', 'proj']`
- panel: `72 Linear + 60 Attention`
- this is an architecture stress test, not an official score and not mixed into Qwen proxy trend audits

| Linear mean | Attention mean | Overall mean | API total (s) | Wall (s) |
|---:|---:|---:|---:|---:|
| 0.519794 | 0.497247 | 0.509545 | 85.571 | 93.427 |

## Error-source decomposition

- Linear interpretation: `unknown`
- Linear W-only/A-only/Both/interaction: `0.000000` / `0.000000` / `0.000000` / `0.000000`
- Attention interpretation: `unknown`
- Attention Q-only/K-only/V-only/QK-only/Both: `0.000000` / `0.000000` / `0.000000` / `0.000000` / `0.000000`

Full per-role/layer/length results are in JSON `case_scores` and `decomposition`.
