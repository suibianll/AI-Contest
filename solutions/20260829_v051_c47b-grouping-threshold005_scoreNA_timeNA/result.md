# C47b CAT grouping 0.5% soft gate（归档，本地接受）

- 日期：2026-08-29
- 版本：v051 / C47b
- 父版本：v050 / C47
- 唯一机制：仅将 CAT-aware grouping 的 operand-local aggregate gate 从 `0.1%` 提高到 `0.5%`；utility、4→64 层次分组、CAT、headroom 与 A@W 静态 Weight 选择不变。
- 合规边界：门控只使用 calibration 的 operand-side量化损失；A@W 仍在 activation_state 冻结后只裁判 Q(W)，不进入 Q(A)。
- 根文件 SHA256：`A209CF4F65ECCA65A73B71A26ED519A31A6F6F8360D06861A6C765601CA639BB`
- 归档文件 SHA256：`A209CF4F65ECCA65A73B71A26ED519A31A6F6F8360D06861A6C765601CA639BB`

## 评测

```powershell
\.venv\Scripts\python.exe -u evaluator\real_model_suite.py --models qwen2.5-0.5b opt-125m gpt2-medium --solution solution.py --candidate-name c47b-grouping-threshold005-highrisk --device cpu --algorithm-device cuda --cache-mode read --seq 128 --calib 2 --test 4 --output artifacts\real_model_suite\c47b-highrisk.json --report logs\evaluations\2026-08-29-c47b-highrisk.md
\.venv\Scripts\python.exe -u evaluator\real_model_suite.py --models gpt2-small pythia-160m --solution solution.py --candidate-name c47b-grouping-threshold005-extra --device cpu --algorithm-device cuda --cache-mode read --seq 128 --calib 2 --test 4 --output artifacts\real_model_suite\c47b-extra.json --report logs\evaluations\2026-08-29-c47b-extra.md
```

| 模型 | Linear | Attention | Total | API |
| --- | ---: | ---: | ---: | ---: |
| GPT-2 small | 133.226930 | 21.120464 | 154.347394 | 55.88s |
| GPT-2 medium | 229.041484 | 43.767156 | 272.808640 | 136.21s |
| OPT-125M | 43.279017 | 19.581565 | 62.860582 | 55.67s |
| Pythia-160M | 138.329016 | 40.614368 | 178.943384 | 54.03s |
| Qwen2.5-0.5B | 286.481992 | 62.862350 | 349.344342 | 149.00s |

- 五模型代理合计：`1018.304342`，较 v048 `1017.984583` 增加 `+0.319759`；较 v050 C47 增加 `+0.251101`。
- OPT 恢复到 v048，GPT-2 small/Attention 持平；所有 API 均小于 300 秒（旧本地代理口径）。
- 修订版官方得分/时间：**`22451 / 234s`**（250 Linear + 200 Attention cases）；
  新版时间限制为 **`420s`（7 分钟）**。本地 official-flow 代理仍只用于相对排序。

## 修订版官方结果

官方在新版 250/200 样例集上确认 C47b 为 `22451 / 234s`，较 v031/C39-FW
和 v034/C41b 的 `21864` 高 `587` 分；耗时仍低于新版 7 分钟限制。

## 结论与下一步

0.5% 的软门过滤了 calibration proxy 中的弱分组候选，显著降低了 OPT 的 `k` 回退，同时保留 Qwen `o` 与 Pythia/medium 的收益。v051/C47b 是新版面板下当前本地官方冠军；下一步可测试 1% 门，但不应引入逐模型硬编码。
