# Cross-model GPT-2 structure probe

- date: 2026-09-01
- purpose: test whether Qwen2.5-0.5B is an unsafe local structural assumption
- source brief: `赛事说明书.txt` (six APIs and data organization; no public model name/shape)
- probe: `evaluator/cross_model_eval.py`
- model: local `models/gpt2`, `GPT2LMHeadModel`, 12 layers, hidden 768, 12 heads, head_dim 64
- attention: fused `attn.c_attn` split by real weight into Q/K/V; MHA (`q_heads=kv_heads`); absolute learned positions; one GELU `ffn_in`; no fabricated gated FFN role
- data: same pinned WikiText revision and calibration/test schedules as `proxy-v2`
- panel: 72 Linear (12 layers × 6 real operations × 1 of five holdout windows) + 60 Attention (12 layers × 5 windows)
- score: same NVFP4/HiF4 reference and candidate APIs; cross-model only, not an official score
- timing: local CUDA API sum only; not convertible to Kunpeng runtime

## Commands

```powershell
.venv\Scripts\python.exe -u evaluator\cross_model_eval.py --model gpt2 --cache-mode read --name root-gpt2 --output artifacts\official_eval\gpt2-root-panel.json --report logs\official_eval\gpt2-root-panel.md
.venv\Scripts\python.exe -u evaluator\cross_model_eval.py --model gpt2 --cache-mode read --solution solutions\20260830_v086_c86-attn-block-final_scoreNA_timeNA\solution.py --name v086-gpt2 --no-decomposition --output artifacts\official_eval\gpt2-v086-panel.json --report logs\official_eval\gpt2-v086-panel.md
.venv\Scripts\python.exe -u evaluator\cross_model_eval.py --model gpt2 --cache-mode read --solution solutions\20260901_v140_linear-roab-pair_rejected\solution.py --name v140-gpt2 --no-decomposition --output artifacts\official_eval\gpt2-v140-panel.json --report logs\official_eval\gpt2-v140-panel.md
.venv\Scripts\python.exe -u evaluator\cross_model_eval.py --model gpt2 --cache-mode read --solution solutions\20260901_v147_v86-attention-v140-linear_rejected\solution.py --name v147-gpt2 --no-decomposition --output artifacts\official_eval\gpt2-v147-panel.json --report logs\official_eval\gpt2-v147-panel.md
```

## Results

| Candidate | Official score | GPT-2 Linear | GPT-2 Attention | GPT-2 overall | local API sum (s) |
|---|---:|---:|---:|---:|---:|
| v86 | 16744 | 0.375010 | 0.411100 | 0.391414 | 102.649 |
| v147 | 16579 | 0.518011 | 0.411100 | 0.469415 | 103.670 |
| v140 | 15838 | 0.519794 | 0.497247 | 0.509545 | 85.571 |

Official order is `v86 > v147 > v140`; GPT-2 order is `v140 > v147 > v86`, so all three
pairwise relations invert. The Qwen panel also ranks v147/v140 above v86. This is evidence that
Qwen structure is not official evidence, but changing only the model does not repair the proxy:
hidden weights/data, case composition, or official implementation details remain unmatched.

The root/v147 detailed decomposition is in `artifacts/official_eval/gpt2-root-panel.json`:
Linear is `paired_coordinate_coupling_likely` (W-only `-240.979`, A-only `-138.412`, Both
`0.518`, interaction `379.909`); Attention is `paired_qk_coupling_likely` (Q-only `-25.810`,
K-only `-28.547`, V-only `0.016`, QK-only `0.389`, Both `0.411`).
