# Cross-model GPT probe

- model: `opt-125m` (`OPTForCausalLM`)
- protocol: `cross-model-probe-v1`; base codec: `proxy-v2`
- layers/hidden/heads: `12/768/12/64` (q/kv)
- roles: `['q', 'k', 'v', 'o', 'ffn_in', 'proj']`
- panel: `72 Linear + 0 Attention`
- this is an architecture stress test, not an official score and not mixed into Qwen proxy trend audits

| Linear mean | Attention mean | Overall mean | API total (s) | Wall (s) |
|---:|---:|---:|---:|---:|
| -0.177596 | 0.000000 | -0.177596 | 110.064 | 115.523 |

## Error-source decomposition

- Linear interpretation: `activation_dominant`
- Linear W-only/A-only/Both/interaction: `-352.335089` / `-186.770746` / `-0.177596` / `538.928240`
- Attention interpretation: `unknown`
- Attention Q-only/K-only/V-only/QK-only/Both: `0.000000` / `0.000000` / `0.000000` / `0.000000` / `0.000000`

### Static Linear role/family gain

| Group | cases | gain |
|---|---:|---:|
| family:fc | 12 | 0.560967 |
| family:o | 12 | 0.384963 |
| family:proj | 12 | -4.005936 |
| family:qkv | 36 | 0.664811 |
| role:ffn_in | 12 | 0.560967 |
| role:k | 12 | 0.729812 |
| role:o | 12 | 0.384963 |
| role:proj | 12 | -4.005936 |
| role:q | 12 | 0.705073 |
| role:v | 12 | 0.559547 |

Static Linear q/k/v are projection roles; the Attention Q/K/V control arms above are a separate dynamic path.
Full per-role/layer/length results are in JSON `case_scores` and `decomposition`.
