# Cross-model GPT probe

- model: `opt-125m` (`OPTForCausalLM`)
- protocol: `cross-model-probe-v1`; base codec: `proxy-v2`
- layers/hidden/heads: `12/768/12/64` (q/kv)
- roles: `['q', 'k', 'v', 'o', 'ffn_in', 'proj']`
- panel: `72 Linear + 60 Attention`
- this is an architecture stress test, not an official score and not mixed into Qwen proxy trend audits

| Linear mean | Attention mean | Overall mean | API total (s) | Wall (s) |
|---:|---:|---:|---:|---:|
| -0.151964 | 0.070971 | -0.050630 | 155.272 | 164.091 |

## Error-source decomposition

- Linear interpretation: `activation_dominant`
- Linear W-only/A-only/Both/interaction: `-352.905608` / `-186.870725` / `-0.151964` / `539.624369`
- Attention interpretation: `paired_qk_coupling_likely`
- Attention Q-only/K-only/V-only/QK-only/Both: `-48.366025` / `-44.011918` / `0.020062` / `0.046376` / `0.070971`

### Static Linear role/family gain

| Group | cases | gain |
|---|---:|---:|
| family:fc | 12 | 0.560338 |
| family:o | 12 | 0.386113 |
| family:proj | 12 | -3.854724 |
| family:qkv | 36 | 0.665496 |
| role:ffn_in | 12 | 0.560338 |
| role:k | 12 | 0.730433 |
| role:o | 12 | 0.386113 |
| role:proj | 12 | -3.854724 |
| role:q | 12 | 0.705637 |
| role:v | 12 | 0.560420 |

Static Linear q/k/v are projection roles; the Attention Q/K/V control arms above are a separate dynamic path.
Full per-role/layer/length results are in JSON `case_scores` and `decomposition`.
