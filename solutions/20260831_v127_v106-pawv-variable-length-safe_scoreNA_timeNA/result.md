# v127 — v106 Linear + PAWV variable-length safe

- Date: 2026-08-31
- Parent: v106 / `solutions/20260830_v106_l2-expansive-cat-active_score294.272633_time413s/`
- Change: replace fixed-size PAWV token Gram accumulation with per-sequence-length
  diagonal states; retain v106 Linear path.
- Source SHA256: `F15E112C7E832D019EE83D707ACD9D72FEF121A306E4CC3B50DBBC2CBB574924`
- Official score/runtime: `NA / NA` (not submitted).

## Canonical v4 sampled result

Command:

```powershell
.\.venv\Scripts\python -u evaluator\real_model_suite.py `
  --models qwen2.5-0.5b --evaluation-profile sampled-means-v1 `
  --sample-layers 8 --sample-test-windows 4 --sample-seed 20260831 `
  --panel-profile qwen-official --device cpu --algorithm-device cpu --cache-mode read `
  --solution solution.py --candidate-name v127-sampled `
  --output artifacts\real_model_suite\v127-sampled-means-qwen.json `
  --report logs\execution\2026-08-31-v127-sampled-means-qwen.md
```

| Metric | Value |
|---|---:|
| Linear mean | `0.509408` (`224` cases) |
| Attention mean | `0.828395` (`32` cases) |
| Local API | `151.136s` (CPU) |
| Wall | `161.840s` |
| Sample layers | `[0,1,5,10,13,15,22,23]` |
| Test windows | `[0,1,2,3]` |
| Roles | `q,k,v,o,fc_gate,fc_up,proj` |

## Legacy full result

The same source was also run with the old full Qwen cache: native total
`419.154521`, legacy panel `294.260802`, Linear mean `0.503458942243`, Attention
mean `0.841980334121`, Local API `453.101930s`, Wall `485.285190s`.
These values are retained for historical comparison only; they are not the v4
primary result and local `453s > 420s` does not imply an official timeout.

## Shape and long-sequence verification

The public Attention calibration shape `[10,128,512,1024,1024]` returned finite,
shape-correct Q/K/V states and exact/unseen-length V lookup. On the same synthetic
input, v127 calibration took `10.873s` versus v74 `0.441s`; long-sequence cost must
be checked separately before any official submission.

Status: `active-research; official-untested`.
