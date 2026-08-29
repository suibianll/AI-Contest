# C65 A@W 折间软混合 0.50（归档，拒绝）

- 日期：2026-08-29
- 版本：v065 / C65
- 父版本：v063 / C63
- 唯一机制：将静态 A@W Q(W) 选择的均值/最差折软混合系数由 `0.25` 提高到 `0.50`；不增加硬门限，Linear 候选 512 行及其余量化路径不变。
- 根文件 SHA256：`8B589C954D181404F6704F3F08418ECA414FB77B2ACCB329640164D60F52FB44`
- 归档文件 SHA256：`8B589C954D181404F6704F3F08418ECA414FB77B2ACCB329640164D60F52FB44`

## 验证

```powershell
\.venv\Scripts\python.exe -m py_compile solution.py
\.venv\Scripts\python.exe -u evaluator\real_model_suite.py --models opt-125m --solution solution.py --candidate-name c65-product-robustmix50-opt --device cpu --algorithm-device cuda --cache-mode read --seq 128 --calib 2 --test 4 --output artifacts\real_model_suite\c65-opt.json --report logs\evaluations\2026-08-29-c65-opt.md
\.venv\Scripts\python.exe -u evaluator\real_model_suite.py --models gpt2-small --solution solution.py --candidate-name c65-product-robustmix50-gpt2small --device cpu --algorithm-device cuda --cache-mode read --seq 128 --calib 2 --test 4 --output artifacts\real_model_suite\c65-gpt2small.json --report logs\evaluations\2026-08-29-c65-gpt2small.md
```

| 模型 | Linear | Attention | Total | API |
| --- | ---: | ---: | ---: | ---: |
| OPT-125M | 65.463078 | 19.581565 | 85.044643 | 55.24s |
| GPT-2 small | 134.255521 | 21.120464 | 155.375985 | 55.80s |

- OPT 与 C63 持平，GPT-2 small 较 C63 `155.436984` 回退 `−0.0610`。
- 官方得分/时间：`NA`；本地 official-flow 代理只用于相对排序。

## 结论

提高最差折权重没有带来稳定收益，保留 C63 的 `robust_mix=0.25`。
