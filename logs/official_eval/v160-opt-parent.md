# Cross-model GPT probe

- model: `opt-125m` (`OPTForCausalLM`)
- protocol: `cross-model-probe-v1`; base codec: `proxy-v2`
- layers/hidden/heads: `12/768/12/64` (q/kv)
- roles: `['q', 'k', 'v', 'o', 'ffn_in', 'proj']`
- panel: `72 Linear + 60 Attention`
- this is an architecture stress test, not an official score and not mixed into Qwen proxy trend audits

| Linear mean | Attention mean | Overall mean | API total (s) | Wall (s) |
|---:|---:|---:|---:|---:|
| 0.059836 | 0.071179 | 0.064992 | 111.998 | 120.999 |

## Error-source decomposition

- Linear interpretation: `paired_coordinate_coupling_likely`
- Linear W-only/A-only/Both/interaction: `-348.664860` / `-186.789507` / `0.059836` / `535.514203`
- Attention interpretation: `paired_qk_coupling_likely`
- Attention Q-only/K-only/V-only/QK-only/Both: `-48.349719` / `-44.006931` / `0.020062` / `0.046499` / `0.071179`

### Static Linear role/family gain

| Group | cases | gain |
|---|---:|---:|
| family:fc | 12 | 0.561079 |
| family:o | 12 | 0.390792 |
| family:proj | 12 | -2.590104 |
| family:qkv | 36 | 0.665750 |
| role:ffn_in | 12 | 0.561079 |
| role:k | 12 | 0.730368 |
| role:o | 12 | 0.390792 |
| role:proj | 12 | -2.590104 |
| role:q | 12 | 0.705139 |
| role:v | 12 | 0.561743 |

Static Linear q/k/v are projection roles; the Attention Q/K/V control arms above are a separate dynamic path.
Full per-role/layer/length results are in JSON `case_scores` and `decomposition`.
