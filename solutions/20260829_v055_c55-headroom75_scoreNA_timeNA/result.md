# C55 Weight headroom 75%（归档，本地接受）

- 日期：2026-08-29
- 版本：v055 / C55
- 父版本：v054 / C54
- 唯一机制：将 post-state FULL64 headroom 覆盖率由 `50%` 调为 `75%`；多折 A@W 仍只在 parent/headroom 静态 Q(W) 之间软选择，CAT/grouping/Q(A) 不变。
- 根文件 SHA256：`8AED9B8E090D328749E64FB224433B562636A5F50125AF05B7875607F2D8E255`
- 归档文件 SHA256：`8AED9B8E090D328749E64FB224433B562636A5F50125AF05B7875607F2D8E255`

## 评测

```powershell
\.venv\Scripts\python.exe -u evaluator\real_model_suite.py --models opt-125m --solution solution.py --candidate-name c55-headroom75-opt --device cpu --algorithm-device cuda --cache-mode read --seq 128 --calib 2 --test 4 --output artifacts\real_model_suite\c55-opt.json --report logs\evaluations\2026-08-29-c55-opt.md
\.venv\Scripts\python.exe -u evaluator\real_model_suite.py --models qwen2.5-0.5b gpt2-medium --solution solution.py --candidate-name c55-headroom75-highrisk --device cpu --algorithm-device cuda --cache-mode read --seq 128 --calib 2 --test 4 --output artifacts\real_model_suite\c55-highrisk.json --report logs\evaluations\2026-08-29-c55-highrisk.md
\.venv\Scripts\python.exe -u evaluator\real_model_suite.py --models gpt2-small pythia-160m --solution solution.py --candidate-name c55-headroom75-extra --device cpu --algorithm-device cuda --cache-mode read --seq 128 --calib 2 --test 4 --output artifacts\real_model_suite\c55-extra.json --report logs\evaluations\2026-08-29-c55-extra.md
```

| 模型 | Linear | Attention | Total | API |
| --- | ---: | ---: | ---: | ---: |
| GPT-2 small | 133.226930 | 21.120464 | 154.347394 | 58.36s |
| GPT-2 medium | 229.041484 | 43.767156 | 272.808640 | 140.79s |
| OPT-125M | 49.823139 | 19.581565 | 69.404704 | 56.48s |
| Pythia-160M | 138.329016 | 40.614368 | 178.943384 | 57.47s |
| Qwen2.5-0.5B | 286.481992 | 62.862350 | 349.344342 | 155.05s |

- 五模型代理合计：`1024.848464`，较 v054 `1023.033335` 增加 `+1.815129`；增量来自 OPT，其余模型逐项持平。
- 所有 API 均小于 300 秒；官方得分/时间：`NA`，本地 official-flow 代理只用于相对排序。

## 结论与下一步

75% 覆盖继续释放了 OPT 的有效 headroom block，A@W layer-level 软选择没有引入其他模型回退。v055 作为当前本地父版本；下一步验证 100% 覆盖的时间与收益上限。
