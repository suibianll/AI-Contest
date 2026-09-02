# Cross-model GPT probe

- model: `gpt2` (`GPT2LMHeadModel`)
- protocol: `cross-model-probe-v1`; base codec: `proxy-v2`
- layers/hidden/heads: `12/768/12/64` (q/kv)
- roles: `['q', 'k', 'v', 'o', 'ffn_in', 'proj']`
- panel: `72 Linear + 60 Attention`
- this is an architecture stress test, not an official score and not mixed into Qwen proxy trend audits

| Linear mean | Attention mean | Overall mean | API total (s) | Wall (s) |
|---:|---:|---:|---:|---:|
| 0.519794 | 0.411100 | 0.470387 | 76.194 | 83.974 |

## Error-source decomposition

- Linear interpretation: `paired_coordinate_coupling_likely`
- Linear W-only/A-only/Both/interaction: `-240.938507` / `-138.398840` / `0.519794` / `379.857141`
- Attention interpretation: `paired_qk_coupling_likely`
- Attention Q-only/K-only/V-only/QK-only/Both: `-25.809794` / `-28.547240` / `0.016191` / `0.389446` / `0.411100`

### Static Linear role/family gain

| Group | cases | gain |
|---|---:|---:|
| family:fc | 12 | 0.458207 |
| family:o | 12 | 0.457787 |
| family:proj | 12 | 0.312901 |
| family:qkv | 36 | 0.629956 |
| role:ffn_in | 12 | 0.458207 |
| role:k | 12 | 0.672904 |
| role:o | 12 | 0.457787 |
| role:proj | 12 | 0.312901 |
| role:q | 12 | 0.633883 |
| role:v | 12 | 0.583081 |

Static Linear q/k/v are projection roles; the Attention Q/K/V control arms above are a separate dynamic path.
Full per-role/layer/length results are in JSON `case_scores` and `decomposition`.
