# C63 Linear 候选评估行数 512（本地接受）

- 日期：2026-08-29
- 版本：v063 / C63
- 父版本：v062 / C62
- 唯一机制：将 `weight_sample` 的 Linear 候选评估预算由 256 行提高到 512 行；实际全量权重量化、在线激活状态、CAT Gram（≤4096 时 1024 行、超宽时 256 行）和 A@W 静态 Q(W) 目标不变。
- 根文件 SHA256：`D58F0F9EC3BDE2A88BDCE12EDA445D8255DBC019E1903006C7C7EAD2CC12A451`
- 归档文件 SHA256：`D58F0F9EC3BDE2A88BDCE12EDA445D8255DBC019E1903006C7C7EAD2CC12A451`

## 验证

```powershell
\.venv\Scripts\python.exe -m py_compile solution.py
\.venv\Scripts\python.exe -u evaluator\real_model_suite.py --models gpt2-small opt-125m qwen2.5-0.5b --solution solution.py --candidate-name c63-linear-weight-eval512-screen --device cpu --algorithm-device cuda --cache-mode read --seq 128 --calib 2 --test 4 --output artifacts\real_model_suite\c63-screen.json --report logs\evaluations\2026-08-29-c63-screen.md
\.venv\Scripts\python.exe -u evaluator\real_model_suite.py --models gpt2-medium pythia-160m --solution solution.py --candidate-name c63-linear-weight-eval512-rest --device cpu --algorithm-device cuda --cache-mode read --seq 128 --calib 2 --test 4 --output artifacts\real_model_suite\c63-rest.json --report logs\evaluations\2026-08-29-c63-rest.md
```

| 模型 | Linear | Attention | Total | API |
| --- | ---: | ---: | ---: | ---: |
| GPT-2 small | 134.316520 | 21.120464 | 155.436984 | 56.03s |
| GPT-2 medium | 231.081080 | 43.767156 | 274.848236 | 137.72s |
| OPT-125M | 65.463078 | 19.581565 | 85.044643 | 54.06s |
| Pythia-160M | 138.282530 | 40.614368 | 178.896898 | 55.33s |
| Qwen2.5-0.5B | 287.007536 | 62.862350 | 349.869886 | 151.69s |
| **五模型合计** |  |  | **1044.096647** |  |

- 相对 C62 五模型合计 `1026.254189` 提升 `+17.842458`；五个模型均正向（GPT‑2 small `+0.648320`、medium `+1.770842`、OPT `+15.192022`、Pythia `+0.084215`、Qwen `+0.147059`）。
- 所有 API 时间均低于 300s；合规路径仍未把 A@W 结果写入在线激活状态。
- 官方得分/时间：`NA`；本地 official-flow 代理只用于相对排序。

## 结论

512 行的 Linear 候选统计显著降低了 256 行排序噪声，尤其改善 OPT；C63 保留为当前本地父版本。后续实验在 C63 上做单变量改动并逐项归档。
