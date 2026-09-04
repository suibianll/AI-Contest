# Cross-model GPT probe

- model: `gpt2` (`GPT2LMHeadModel`)
- protocol: `cross-model-probe-v1`; base codec: `proxy-v2`
- layers/hidden/heads: `12/768/12/64` (q/kv)
- roles: `['q', 'k', 'v', 'o', 'ffn_in', 'proj']`
- panel: `72 Linear + 60 Attention`
- this is an architecture stress test, not an official score and not mixed into Qwen proxy trend audits

| Linear mean | Attention mean | Overall mean | API total (s) | Wall (s) |
|---:|---:|---:|---:|---:|
| 0.605071 | 0.384798 | 0.504947 | 158.382 | 167.242 |

## Error-source decomposition

- Linear interpretation: `paired_coordinate_coupling_likely`
- Linear W-only/A-only/Both/interaction: `-209.514594` / `-142.105728` / `0.605071` / `352.225393`
- Attention interpretation: `paired_qk_coupling_likely`
- Attention Q-only/K-only/V-only/QK-only/Both: `-27.380358` / `-29.057631` / `0.016191` / `0.362649` / `0.384798`

### Static Linear role/family gain

| Group | cases | gain |
|---|---:|---:|
| family:fc | 12 | 0.514235 |
| family:o | 12 | 0.518600 |
| family:proj | 12 | 0.507559 |
| family:qkv | 36 | 0.696677 |
| role:ffn_in | 12 | 0.514235 |
| role:k | 12 | 0.759281 |
| role:o | 12 | 0.518600 |
| role:proj | 12 | 0.507559 |
| role:q | 12 | 0.696729 |
| role:v | 12 | 0.634020 |

Static Linear q/k/v are projection roles; the Attention Q/K/V control arms above are a separate dynamic path.
Full per-role/layer/length results are in JSON `case_scores` and `decomposition`.
