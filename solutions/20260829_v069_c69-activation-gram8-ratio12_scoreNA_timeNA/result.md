# C69 激活二次项 Gram-8 覆盖比例 12%（归档，本地接受）

- 日期：2026-08-29
- 版本：v069 / C69
- 父版本：v066 / C66
- 唯一机制：将激活二次项 Gram-8 细化的动态覆盖上限由 `8%` 调到 `12%`；Linear 512 行统计、CAT 宽度分流、静态 A@W Q(W) 选择和在线路径不变。
- 根文件 SHA256：`1F71CA11FA9707EB9720438EC6D780CC6F520FBA80437B3215398608D5866CA1`
- 归档文件 SHA256：`1F71CA11FA9707EB9720438EC6D780CC6F520FBA80437B3215398608D5866CA1`

## 验证

```powershell
\.venv\Scripts\python.exe -m py_compile solution.py
\.venv\Scripts\python.exe -u evaluator\real_model_suite.py --models gpt2-small opt-125m qwen2.5-0.5b --solution solution.py --candidate-name c69-activation-gram8-ratio12-screen --device cpu --algorithm-device cuda --cache-mode read --seq 128 --calib 2 --test 4 --output artifacts\real_model_suite\c69-screen.json --report logs\evaluations\2026-08-29-c69-screen.md
\.venv\Scripts\python.exe -u evaluator\real_model_suite.py --models gpt2-medium pythia-160m --solution solution.py --candidate-name c69-activation-gram8-ratio12-rest --device cpu --algorithm-device cuda --cache-mode read --seq 128 --calib 2 --test 4 --output artifacts\real_model_suite\c69-rest.json --report logs\evaluations\2026-08-29-c69-rest.md
```

| 模型 | Linear | Attention | Total | API |
| --- | ---: | ---: | ---: | ---: |
| GPT-2 small | 134.329831 | 21.306236 | 155.636068 | 56.94s |
| GPT-2 medium | 231.098280 | 43.760024 | 274.858304 | 138.83s |
| OPT-125M | 65.472699 | 19.647602 | 85.120301 | 54.95s |
| Pythia-160M | 138.291865 | 40.647879 | 178.939745 | 55.77s |
| Qwen2.5-0.5B | 287.032704 | 63.119717 | 350.152420 | 156.68s |

- 五模型合计：`1044.706838`，较 C66 `1044.703044` 提升 `+0.003794`；五个模型均未回退（GPT-2 small `+0.002492`、OPT `+0.000420`、Pythia `+0.000882`，medium/Qwen 持平）。
- 五次 API 总耗时均低于 300 秒；官方得分/时间：`NA`，本地 official-flow 代理只用于相对排序。

## 结论

12% 覆盖上限在五模型上保持非负并取得微小净增，接受为当前根版本；后续更大比例需重新做全模型回归，避免为局部收益牺牲跨模型稳定性。
