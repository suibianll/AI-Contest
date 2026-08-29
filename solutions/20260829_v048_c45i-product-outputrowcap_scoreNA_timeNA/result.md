# C45i 按输出行数控制的静态 A@W 产品选择（归档，当前父版本）

- 日期：2026-08-29
- 版本：v048 / C45i
- 父版本：v043 / C45f
- 唯一机制：保留 post-state 的全部多折 Weight headroom，但将 A@W 静态产品选择限制为 `out_features <= 4096`。因此 Qwen 的 4864 行 `fc_gate/fc_up` 不参与高方差候选，896 行 `proj` 仍可参与；在线 `activation_state` 与 Q(A) 路径不变。
- 合规边界：A@W 只用于离线静态 `weight_params`，不写入 `activation_state`，不参与在线激活量化；无硬性逐 fold 否决门。
- 根文件 SHA256：`15A7579485856FF330CB9A42AA368DF1C52F95EDEADBD748247B1C0C35F48CCE`
- 归档文件 SHA256：`15A7579485856FF330CB9A42AA368DF1C52F95EDEADBD748247B1C0C35F48CCE`

## 评测

```powershell
\.venv\Scripts\python.exe -u evaluator\real_model_suite.py --models qwen2.5-0.5b --solution solution.py --candidate-name c45i-product-outputrowcap --device cpu --algorithm-device cuda --cache-mode read --seq 128 --calib 2 --test 4 --output artifacts\real_model_suite\c45i-qwen.json --report logs\evaluations\2026-08-29-c45i-qwen.md
\.venv\Scripts\python.exe -u evaluator\real_model_suite.py --models gpt2-small opt-125m --solution solution.py --candidate-name c45i-product-outputrowcap-2model --device cpu --algorithm-device cuda --cache-mode read --seq 128 --calib 2 --test 4 --output artifacts\real_model_suite\c45i-2model.json --report logs\evaluations\2026-08-29-c45i-2model.md
\.venv\Scripts\python.exe -u evaluator\real_model_suite.py --models gpt2-medium pythia-160m --solution solution.py --candidate-name c45i-product-outputrowcap-extra --device cpu --algorithm-device cuda --cache-mode read --seq 128 --calib 2 --test 4 --output artifacts\real_model_suite\c45i-extra.json --report logs\evaluations\2026-08-29-c45i-extra.md
```

| 模型 | Linear | Attention | Total | API |
| --- | ---: | ---: | ---: | ---: |
| GPT-2 small | 133.226930 | 21.120464 | 154.347394 | 46.10s |
| GPT-2 medium | 229.019937 | 43.767156 | 272.787093 | 117.79s |
| OPT-125M | 43.279017 | 19.581565 | 62.860582 | 44.21s |
| Pythia-160M | 138.246673 | 40.614368 | 178.861041 | 47.79s |
| Qwen2.5-0.5B | 286.266123 | 62.862350 | 349.128473 | 124.82s |

- 五模型代理合计：`1017.984583`，较 C45f `1017.892499` 增加 `+0.092084`；除 Qwen 外四模型逐项持平。
- 所有模型 `official_api_total_seconds < 300s`，本地流程有效。数据集、窗口、model revision 和 codec SHA256 详见三个评估 JSON/report；缓存使用 `schema1` 固定 WikiText-2 窗口。
- 官方得分/时间：`NA`；本地 official-flow 代理只用于相对排序。

## 结论与下一步

C45i 是当前本地父版本：在不收紧全局防御门的前提下，精确跳过已验证会回退的 Qwen 4864 行 FFN 产品候选，同时保留 896 行 `proj` 的收益。下一步只接受能在该五模型面板上带来可复现正增量、且不把 A@W 引入 Q(A) 的小步 Weight/Linear 变体。
