# C57 A@W 产品选择 block 比例 25%（归档，拒绝）

- 日期：2026-08-29
- 版本：v057 / C57
- 父版本：v056 / C56
- 唯一机制：静态 A@W 产品选择的 block 更新比例由 `12.5%` 提高到 `25%`；全覆盖 headroom、CAT/grouping、在线 Q(A) 均不变。
- 合规边界：A@W 只在 activation_state 冻结后评价少量静态 Q(W) 候选，不进入 activation_state。
- 根文件 SHA256：`E869FC80B0BCEF3313CE4D541B5ED5A841F106E0BFC91FDDC26F8720D1E7AEA5`
- 归档文件 SHA256：`E869FC80B0BCEF3313CE4D541B5ED5A841F106E0BFC91FDDC26F8720D1E7AEA5`

## 评测

```powershell
\.venv\Scripts\python.exe -u evaluator\real_model_suite.py --models opt-125m --solution solution.py --candidate-name c57-product-ratio25-opt --device cpu --algorithm-device cuda --cache-mode read --seq 128 --calib 2 --test 4 --output artifacts\real_model_suite\c57-opt.json --report logs\evaluations\2026-08-29-c57-opt.md
\.venv\Scripts\python.exe -u evaluator\real_model_suite.py --models qwen2.5-0.5b gpt2-medium --solution solution.py --candidate-name c57-product-ratio25-highrisk --device cpu --algorithm-device cuda --cache-mode read --seq 128 --calib 2 --test 4 --output artifacts\real_model_suite\c57-highrisk.json --report logs\evaluations\2026-08-29-c57-highrisk.md
```

| 模型 | Linear | Attention | Total | API |
| --- | ---: | ---: | ---: | ---: |
| OPT-125M | 49.987750 | 19.581565 | 69.569316 | 62.12s |
| Qwen2.5-0.5B | 286.032560 | 62.862350 | 348.894910 | 171.52s |
| GPT-2 medium | 228.673780 | 43.767156 | 272.440936 | 156.14s |

- 三个已运行模型均低于 v056：OPT `-0.319782`、Qwen `-0.449432`、GPT-2 medium `-0.474983`。
- 官方得分/时间：`NA`；本地 official-flow 代理只用于相对排序。

## 结论

扩大静态候选容量导致跨模型过拟合，A@W 选择器无法弥补候选生成自由度；v057 拒绝，恢复 12.5% 比例。
