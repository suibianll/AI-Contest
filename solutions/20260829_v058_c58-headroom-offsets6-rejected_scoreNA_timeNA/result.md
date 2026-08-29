# C58 Weight headroom E6M2 offsets ±6（归档，拒绝）

- 日期：2026-08-29
- 版本：v058 / C58
- 父版本：v056 / C56
- 唯一机制：将 headroom scale beam 从 `{-4,…,4}` 扩展到 `{-6,…,6}`，仍由 full-H 选 top-4、再用多折 A@W 只裁判静态 Q(W)。
- 根文件 SHA256：`E2937B5A18A93E86465CD82DD24CB4F9FEED10F9C623916D525B2E012DAB9513`
- 归档文件 SHA256：`E2937B5A18A93E86465CD82DD24CB4F9FEED10F9C623916D525B2E012DAB9513`

## 评测

```powershell
\.venv\Scripts\python.exe -m py_compile solution.py
\.venv\Scripts\python.exe -u evaluator\real_model_suite.py --models opt-125m --solution solution.py --candidate-name c58-headroom-offsets6-opt --device cpu --algorithm-device cuda --cache-mode read --seq 128 --calib 2 --test 4 --output artifacts\real_model_suite\c58-opt.json --report logs\evaluations\2026-08-29-c58-opt.md
\.venv\Scripts\python.exe -u evaluator\real_model_suite.py --models qwen2.5-0.5b --solution solution.py --candidate-name c58-headroom-offsets6-qwen --device cpu --algorithm-device cuda --cache-mode read --seq 128 --calib 2 --test 4 --output artifacts\real_model_suite\c58-qwen.json --report logs\evaluations\2026-08-29-c58-qwen.md
```

| 模型 | Linear | Attention | Total | API |
| --- | ---: | ---: | ---: | ---: |
| OPT-125M | 50.307481 | 19.581565 | 69.889046 | 54.18s |
| Qwen2.5-0.5B | 286.481992 | 62.862350 | 349.344342 | 154.15s |

- OPT 与 Qwen 均与 v056 在报告精度内持平（OPT 差约 `−5.2e-5`），没有收益。
- 官方得分/时间：`NA`；本地 official-flow 代理只用于相对排序。

## 结论

额外 E6M2 邻域在 full-H top-4 后没有形成有效静态候选，只增加极小开销；v058 拒绝并恢复 `{-4,…,4}`。
