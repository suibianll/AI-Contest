# C62 CAT WᵀW 宽度分流（本地接受）

- 日期：2026-08-29
- 版本：v062 / C62
- 父版本：v056 / C56（C61 rejected）
- 唯一机制：CAT-64 候选的 operand-local `WᵀW` 统计在输入通道 `≤4096` 时使用最多 1024 个确定性输出行；`>4096` 时回退原 256 行统计。候选指标、在线状态、A@W 静态 Q(W) 选择和门限均不变。
- 根文件 SHA256：`7CA0F8A13496218FE6AC82EDAF018EAA533C93C3BC500085A0CB993D7C08F2DB`
- 归档文件 SHA256：`7CA0F8A13496218FE6AC82EDAF018EAA533C93C3BC500085A0CB993D7C08F2DB`

## 验证

```powershell
\.venv\Scripts\python.exe -m py_compile solution.py
\.venv\Scripts\python.exe -u evaluator\real_model_suite.py --models qwen2.5-0.5b --solution solution.py --candidate-name c62-cat-weightgram-widthcap-qwen --device cpu --algorithm-device cuda --cache-mode read --seq 128 --calib 2 --test 4 --output artifacts\real_model_suite\c62-qwen.json --report logs\evaluations\2026-08-29-c62-qwen.md
\.venv\Scripts\python.exe -u evaluator\real_model_suite.py --models gpt2-small gpt2-medium opt-125m pythia-160m --solution solution.py --candidate-name c62-cat-weightgram-widthcap-rest --device cpu --algorithm-device cuda --cache-mode read --seq 128 --calib 2 --test 4 --output artifacts\real_model_suite\c62-rest.json --report logs\evaluations\2026-08-29-c62-rest.md
```

| 模型 | Linear | Attention | Total | API |
| --- | ---: | ---: | ---: | ---: |
| GPT-2 small | 133.668200 | 21.120464 | 154.788664 | 56.54s |
| GPT-2 medium | 229.310238 | 43.767156 | 273.077394 | 137.87s |
| OPT-125M | 50.271056 | 19.581565 | 69.852621 | 54.28s |
| Pythia-160M | 138.198315 | 40.614368 | 178.812683 | 55.64s |
| Qwen2.5-0.5B | 286.860477 | 62.862350 | 349.722827 | 152.35s |
| **五模型合计** |  |  | **1026.254189** |  |

- 五模型合计较 v056 `1025.440137` 提升 `+0.814052`；GPT-2 small、GPT-2 medium、Qwen 提升，OPT/Pythia 轻微回退。
- 回归/合规测试：C61 已验证 55 tests passed；C62 仅增加宽度条件，不改 API/在线状态。
- 官方得分/时间：`NA`；本地 official-flow 代理只用于相对排序。

## 结论

C62 消除了 C61 在 Qwen 4864-wide `proj` 上的结构性回退，并取得当前五模型最佳本地合计。保留为当前根父版本，后续实验必须以 C62 为基线并逐候选归档。
