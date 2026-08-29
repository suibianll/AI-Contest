# C67 Linear 候选评估行数 640（归档，拒绝）

- 日期：2026-08-29
- 版本：v067 / C67
- 父版本：v066 / C66
- 唯一机制：将 Linear 候选排序的确定性 `weight_sample` 预算由 512 行提高到 640 行；其余量化与在线路径不变。
- 根文件 SHA256：`A0621F2EEC198C5FC569443566806F1F65D7CBCE63B58DF0B0FF3E16702780A4`
- 归档文件 SHA256：`A0621F2EEC198C5FC569443566806F1F65D7CBCE63B58DF0B0FF3E16702780A4`

## 验证

```powershell
\.venv\Scripts\python.exe -m py_compile solution.py
\.venv\Scripts\python.exe -u evaluator\real_model_suite.py --models gpt2-small opt-125m qwen2.5-0.5b --solution solution.py --candidate-name c67-linear-weight-eval640-screen --device cpu --algorithm-device cuda --cache-mode read --seq 128 --calib 2 --test 4 --output artifacts\real_model_suite\c67-screen.json --report logs\evaluations\2026-08-29-c67-screen.md
\.venv\Scripts\python.exe -u evaluator\real_model_suite.py --models gpt2-medium pythia-160m --solution solution.py --candidate-name c67-linear-weight-eval640-rest --device cpu --algorithm-device cuda --cache-mode read --seq 128 --calib 2 --test 4 --output artifacts\real_model_suite\c67-rest.json --report logs\evaluations\2026-08-29-c67-rest.md
```

| 模型 | Linear | Attention | Total | API |
| --- | ---: | ---: | ---: | ---: |
| GPT-2 small | 134.335127 | 21.306236 | 155.641364 | 56.34s |
| GPT-2 medium | 230.953064 | 43.760024 | 274.713088 | 140.94s |
| OPT-125M | 65.687656 | 19.647602 | 85.335258 | 54.00s |
| Pythia-160M | 138.183341 | 40.647879 | 178.831220 | 57.70s |
| Qwen2.5-0.5B | 287.110685 | 63.119717 | 350.230401 | 155.95s |
| **五模型合计** |  |  | **1044.751331** |  |

- 合计较 C66 `1044.703044` 仅 `+0.048287`；GPT‑2 medium `−0.145216`、Pythia `−0.107643`，跨模型收益不稳。
- 官方得分/时间：`NA`；本地 official-flow 代理只用于相对排序。

## 结论

640 行处于 512 与 1024 的窄平台，净提升不足且牺牲两模型；恢复 C66 的 512 行预算。
