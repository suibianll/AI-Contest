# Cross-model GPT probe

- model: `gpt2` (`GPT2LMHeadModel`)
- protocol: `cross-model-probe-v1`; base codec: `proxy-v2`
- layers/hidden/heads: `12/768/12/64` (q/kv)
- roles: `['q', 'k', 'v', 'o', 'ffn_in', 'proj']`
- panel: `72 Linear + 0 Attention`
- this is an architecture stress test, not an official score and not mixed into Qwen proxy trend audits

| Linear mean | Attention mean | Overall mean | API total (s) | Wall (s) |
|---:|---:|---:|---:|---:|
| 0.603900 | 0.000000 | 0.603900 | 111.710 | 116.983 |

## Error-source decomposition

- Linear interpretation: `paired_coordinate_coupling_likely`
- Linear W-only/A-only/Both/interaction: `-209.587762` / `-142.097466` / `0.603900` / `352.289128`
- Attention interpretation: `unknown`
- Attention Q-only/K-only/V-only/QK-only/Both: `0.000000` / `0.000000` / `0.000000` / `0.000000` / `0.000000`

### Static Linear role/family gain

| Group | cases | gain |
|---|---:|---:|
| family:fc | 12 | 0.514416 |
| family:o | 12 | 0.512606 |
| family:proj | 12 | 0.504719 |
| family:qkv | 36 | 0.697220 |
| role:ffn_in | 12 | 0.514416 |
| role:k | 12 | 0.760920 |
| role:o | 12 | 0.512606 |
| role:proj | 12 | 0.504719 |
| role:q | 12 | 0.696655 |
| role:v | 12 | 0.634086 |

Static Linear q/k/v are projection roles; the Attention Q/K/V control arms above are a separate dynamic path.
Full per-role/layer/length results are in JSON `case_scores` and `decomposition`.
