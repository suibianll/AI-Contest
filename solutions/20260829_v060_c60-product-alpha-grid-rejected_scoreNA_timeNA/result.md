# C60 A@W 产品条件步长扩展（归档，拒绝）

- 日期：2026-08-29
- 版本：v060 / C60
- 父版本：v056 / C56
- 唯一机制：将静态 A@W 条件更新步长由 `{0.10,0.25,0.50}` 扩展为 `{0.05,0.10,0.25,0.50,0.75}`；headroom、CAT/grouping、在线 Q(A) 不变。
- 根文件 SHA256：`62DB156B032A718248E55E40C675919709568BE306DE271B150E8A70288B4A77`
- 归档文件 SHA256：`62DB156B032A718248E55E40C675919709568BE306DE271B150E8A70288B4A77`

## 评测

```powershell
\.venv\Scripts\python.exe -m py_compile solution.py
\.venv\Scripts\python.exe -u evaluator\real_model_suite.py --models opt-125m --solution solution.py --candidate-name c60-product-alpha-grid-opt --device cpu --algorithm-device cuda --cache-mode read --seq 128 --calib 2 --test 4 --output artifacts\real_model_suite\c60-opt.json --report logs\evaluations\2026-08-29-c60-opt.md
```

| 模型 | Linear | Attention | Total | API |
| --- | ---: | ---: | ---: | ---: |
| OPT-125M | 48.878251 | 19.581565 | 68.459816 | 57.82s |

- OPT 从 v056 的 `69.889098` 回退 `−1.429282`，未继续运行其他模型。
- 官方得分/时间：`NA`；本地 official-flow 代理只用于相对排序。

## 结论

扩大条件步长网格引入额外连续自由度并损害 OPT；v060 拒绝，恢复三档 `{0.10,0.25,0.50}`。
