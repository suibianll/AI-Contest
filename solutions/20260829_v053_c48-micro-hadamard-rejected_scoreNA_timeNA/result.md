# C48 CAT + micro-Hadamard（归档，拒绝）

- 日期：2026-08-29
- 版本：v053 / C48
- 父版本：v051 / C47b
- 唯一机制：在已选 CAT-64 后尝试 16/32-channel signed Hadamard 组合（每层 4 个确定性候选），只用 operand-local HiF4 loss 软裁判；未改 A@W 选择器或在线状态结构。
- 根文件 SHA256：`56A912452F4B4E6058801E4CC94B92BB9C0F3816B21837257FAC8EEE089A6777`
- 归档文件 SHA256：`56A912452F4B4E6058801E4CC94B92BB9C0F3816B21837257FAC8EEE089A6777`

## 评测

```powershell
\.venv\Scripts\python.exe -u evaluator\real_model_suite.py --models gpt2-small --solution solution.py --candidate-name c48-micro-hadamard-gpt2small --device cpu --algorithm-device cuda --cache-mode read --seq 128 --calib 2 --test 4 --output artifacts\real_model_suite\c48-gpt2small.json --report logs\evaluations\2026-08-29-c48-gpt2small.md
\.venv\Scripts\python.exe -u evaluator\real_model_suite.py --models qwen2.5-0.5b opt-125m --solution solution.py --candidate-name c48-micro-hadamard-highrisk --device cpu --algorithm-device cuda --cache-mode read --seq 128 --calib 2 --test 4 --output artifacts\real_model_suite\c48-highrisk.json --report logs\evaluations\2026-08-29-c48-highrisk.md
```

| 模型 | Linear | Attention | Total | API |
| --- | ---: | ---: | ---: | ---: |
| GPT-2 small | 133.226930 | 21.120464 | 154.347394 | 55.02s |
| Qwen2.5-0.5B | 286.481992 | 62.862350 | 349.344342 | 150.69s |
| OPT-125M | 43.279017 | 19.581565 | 62.860582 | 53.83s |

- 三模型均与 v051 逐项相同，未产生任何 micro-Hadamard 候选替换；只增加约 0–2 秒 API 时间。
- 官方得分/时间：`NA`；本地 official-flow 代理只用于相对排序。

## 结论

在当前 CAT-64 与 grouping 父版本上，16/32 微 Hadamard 组合没有 operand-side 正收益；拒绝并恢复 v051 的轻量路径。
