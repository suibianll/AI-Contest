# C54 Weight headroom 50%（归档，本地接受）

- 日期：2026-08-29
- 版本：v054 / C54
- 父版本：v051 / C47b
- 唯一机制：将 post-state FULL64 headroom 候选的 block 覆盖率由 `25%` 调为 `50%`；每层仍由多折 A@W 软目标在 parent/headroom 两个静态 Q(W) 之间选择，CAT/grouping/Q(A) 不变。
- 合规边界：A@W 只在 activation_state 冻结后评价静态 `weight_params`，不影响激活状态、覆盖率选择或在线量化。
- 根文件 SHA256：`D1BEB9ADD6C575D3522A093C24E9BADA82524CDF56967915651A1F42F5012ED6`
- 归档文件 SHA256：`D1BEB9ADD6C575D3522A093C24E9BADA82524CDF56967915651A1F42F5012ED6`

## 评测

```powershell
\.venv\Scripts\python.exe -u evaluator\real_model_suite.py --models gpt2-small opt-125m --solution solution.py --candidate-name c54-headroom50-2model --device cpu --algorithm-device cuda --cache-mode read --seq 128 --calib 2 --test 4 --output artifacts\real_model_suite\c54-2model.json --report logs\evaluations\2026-08-29-c54-2model.md
\.venv\Scripts\python.exe -u evaluator\real_model_suite.py --models qwen2.5-0.5b gpt2-medium --solution solution.py --candidate-name c54-headroom50-highrisk --device cpu --algorithm-device cuda --cache-mode read --seq 128 --calib 2 --test 4 --output artifacts\real_model_suite\c54-highrisk.json --report logs\evaluations\2026-08-29-c54-highrisk.md
\.venv\Scripts\python.exe -u evaluator\real_model_suite.py --models pythia-160m --solution solution.py --candidate-name c54-headroom50-pythia --device cpu --algorithm-device cuda --cache-mode read --seq 128 --calib 2 --test 4 --output artifacts\real_model_suite\c54-pythia.json --report logs\evaluations\2026-08-29-c54-pythia.md
```

| 模型 | Linear | Attention | Total | API |
| --- | ---: | ---: | ---: | ---: |
| GPT-2 small | 133.226930 | 21.120464 | 154.347394 | 55.14s |
| GPT-2 medium | 229.041484 | 43.767156 | 272.808640 | 134.30s |
| OPT-125M | 48.008010 | 19.581565 | 67.589575 | 53.39s |
| Pythia-160M | 138.329016 | 40.614368 | 178.943384 | 56.58s |
| Qwen2.5-0.5B | 286.481992 | 62.862350 | 349.344342 | 149.42s |

- 五模型代理合计：`1023.033335`，较 v051 `1018.304342` 增加 `+4.728993`；增量全部来自 OPT，其他四模型逐项持平。
- 所有 API 均小于 300 秒；官方得分/时间：`NA`，本地 official-flow 代理只用于相对排序。

## 结论与下一步

50% headroom 覆盖在 A@W 静态裁判下安全释放了 OPT 的有效 block，收益远高于 CAT 微组合。v054 作为当前本地父版本；下一步只测试 75% 覆盖，若收益趋于饱和则保留 50% 以控制时间。
