# C47c CAT grouping 1% soft gate（归档，拒绝）

- 日期：2026-08-29
- 版本：v052 / C47c
- 父版本：v051 / C47b
- 唯一机制：仅将 CAT-aware grouping 的 operand-local aggregate gate 从 `0.5%` 调整为 `1.0%`；其他算法和状态格式不变。
- 根文件 SHA256：`F3CD3AF8CD5ACBF526B0E9B786DE65AB87B7893BB7417EC0C7FC9B39A2038C22`
- 归档文件 SHA256：`F3CD3AF8CD5ACBF526B0E9B786DE65AB87B7893BB7417EC0C7FC9B39A2038C22`

## 评测

```powershell
\.venv\Scripts\python.exe -u evaluator\real_model_suite.py --models qwen2.5-0.5b opt-125m gpt2-medium --solution solution.py --candidate-name c47c-grouping-threshold01-highrisk --device cpu --algorithm-device cuda --cache-mode read --seq 128 --calib 2 --test 4 --output artifacts\real_model_suite\c47c-highrisk.json --report logs\evaluations\2026-08-29-c47c-highrisk.md
\.venv\Scripts\python.exe -u evaluator\real_model_suite.py --models gpt2-small pythia-160m --solution solution.py --candidate-name c47c-grouping-threshold01-extra --device cpu --algorithm-device cuda --cache-mode read --seq 128 --calib 2 --test 4 --output artifacts\real_model_suite\c47c-extra.json --report logs\evaluations\2026-08-29-c47c-extra.md
```

| 模型 | Linear | Attention | Total | API |
| --- | ---: | ---: | ---: | ---: |
| GPT-2 small | 133.226930 | 21.120464 | 154.347394 | 55.73s |
| GPT-2 medium | 229.019937 | 43.767156 | 272.787093 | 131.87s |
| OPT-125M | 43.279017 | 19.581565 | 62.860582 | 52.64s |
| Pythia-160M | 138.246673 | 40.614368 | 178.861041 | 53.48s |
| Qwen2.5-0.5B | 286.481992 | 62.862350 | 349.344342 | 148.29s |

- 五模型代理合计：`1018.200452`，相对 v048 增加 `+0.215869`；由于四个非 Qwen 模型回到 v048，整体低于 v051 的 `1018.304342`。
- 官方得分/时间：`NA`；本地 official-flow 代理只用于相对排序。

## 结论

1% 门过滤掉了 medium/Pythia 的小幅分组收益，Qwen 没有进一步变化；候选净收益低于 v051，因此拒绝并恢复 `0.5%` 门。
