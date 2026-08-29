# C47 CAT-aware channel grouping（归档，本地接受）

- 日期：2026-08-29
- 版本：v050 / C47
- 父版本：v048 / C45i
- 唯一机制：在已选 SmoothQuant 坐标上，用 operand-only utility `|AᵀA|√diag(WᵀW) + |WᵀW|√diag(AᵀA)` 做 4→8→16→32→64 层次分组，再将候选置换送入现有 CAT-64；协方差最多 2048 通道，使用软均值/尾部 gate。
- 合规边界：只使用 `AᵀA`、`WᵀW` 和标准 HiF4 operand loss；A@W 产品选择仍仅发生在 activation_state 冻结后的静态 Q(W)，不影响 Q(A)。
- 根文件 SHA256：`C388AFC5BA18106DFCA8057A3D367B840D43DF451873E2A01EABA561426FC1E8`
- 归档文件 SHA256：`C388AFC5BA18106DFCA8057A3D367B840D43DF451873E2A01EABA561426FC1E8`

## 评测

```powershell
\.venv\Scripts\python.exe -u evaluator\real_model_suite.py --models gpt2-small --solution solution.py --candidate-name c47-cat-grouping-gpt2small --device cpu --algorithm-device cuda --cache-mode read --seq 128 --calib 2 --test 4 --output artifacts\real_model_suite\c47-gpt2small.json --report logs\evaluations\2026-08-29-c47-gpt2small.md
\.venv\Scripts\python.exe -u evaluator\real_model_suite.py --models qwen2.5-0.5b gpt2-medium --solution solution.py --candidate-name c47-cat-grouping-highrisk --device cpu --algorithm-device cuda --cache-mode read --seq 128 --calib 2 --test 4 --output artifacts\real_model_suite\c47-highrisk.json --report logs\evaluations\2026-08-29-c47-highrisk.md
\.venv\Scripts\python.exe -u evaluator\real_model_suite.py --models opt-125m pythia-160m --solution solution.py --candidate-name c47-cat-grouping-extra --device cpu --algorithm-device cuda --cache-mode read --seq 128 --calib 2 --test 4 --output artifacts\real_model_suite\c47-extra.json --report logs\evaluations\2026-08-29-c47-extra.md
```

| 模型 | Linear | Attention | Total | API |
| --- | ---: | ---: | ---: | ---: |
| GPT-2 small | 133.226930 | 21.120464 | 154.347394 | 57.49s |
| GPT-2 medium | 229.041484 | 43.767156 | 272.808640 | 133.22s |
| OPT-125M | 42.981262 | 19.581565 | 62.562827 | 53.87s |
| Pythia-160M | 138.382957 | 40.614368 | 178.997325 | 54.31s |
| Qwen2.5-0.5B | 286.474705 | 62.862350 | 349.337055 | 156.89s |

- 五模型代理合计：`1018.053241`，较 v048 `1017.984583` 增加 `+0.068658`；Linear 合计净增 `+0.068658`，Attention 逐项不变。
- OPT Total 回退 `-0.297755`（约 0.47%），但其余模型提升/持平，中位数增量为正；所有 API 均低于 300 秒。
- 官方得分/时间：`NA`；本地 official-flow 代理只用于相对排序。

## 结论与下一步

C47 保留为本地接受候选：分组确实改善 Qwen `o`、Pythia `q/v/fc` 等困难块，且没有触碰在线激活格式。下一步测试更稳的增益门/交叉校准版本，目标是在不牺牲 Qwen 分组收益的情况下减少 OPT `k/v` 回退。
