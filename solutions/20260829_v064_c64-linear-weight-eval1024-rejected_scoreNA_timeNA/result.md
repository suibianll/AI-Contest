# C64 Linear 候选评估行数 1024（归档，拒绝）

- 日期：2026-08-29
- 版本：v064 / C64
- 父版本：v063 / C63
- 唯一机制：将 `weight_sample` 的 Linear 候选评估预算由 512 行提高到 1024 行；其余 CAT 宽度分流、实际量化、在线状态及 A@W Q(W) 选择不变。
- 根文件 SHA256：`63FDD1E6E7EEA03E07E6325950E20193FF9450AA8653AC50DB305AA7C2201C16`
- 归档文件 SHA256：`63FDD1E6E7EEA03E07E6325950E20193FF9450AA8653AC50DB305AA7C2201C16`

## 验证

```powershell
\.venv\Scripts\python.exe -m py_compile solution.py
\.venv\Scripts\python.exe -u evaluator\real_model_suite.py --models gpt2-small opt-125m qwen2.5-0.5b --solution solution.py --candidate-name c64-linear-weight-eval1024-screen --device cpu --algorithm-device cuda --cache-mode read --seq 128 --calib 2 --test 4 --output artifacts\real_model_suite\c64-screen.json --report logs\evaluations\2026-08-29-c64-screen.md
```

| 模型 | Linear | Attention | Total | API |
| --- | ---: | ---: | ---: | ---: |
| GPT-2 small | 134.064366 | 21.120464 | 155.184830 | 56.12s |
| OPT-125M | 63.119458 | 19.581565 | 82.701023 | 54.80s |
| Qwen2.5-0.5B | 287.281990 | 62.862350 | 350.144340 | 155.17s |

- 三模型小计 `588.030193`，较 C63 同三模型 `590.351513` 回退 `−2.321320`；GPT‑2 small/OPT 均回退，Qwen 单独提升。
- 官方得分/时间：`NA`；本地 official-flow 代理只用于相对排序。

## 结论

1024 行候选排序在有限校准预算下方差再次增大，收益不如 C63 的 512 行。未继续评估其余模型，恢复 C63 的 512 行设置。
