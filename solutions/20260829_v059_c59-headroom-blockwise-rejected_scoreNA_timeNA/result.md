# C59 逐 64-block A@W headroom 混合（归档，拒绝）

- 日期：2026-08-29
- 版本：v059 / C59
- 父版本：v056 / C56
- 唯一机制：在全覆盖 headroom 生成后，尝试按 64-channel block 逐块用 A@W 贪心合并 parent/headroom；原有整层 A@W selector 仍保留，在线 state 不变。
- 合规边界：产品只用于静态 Q(W) block 选择，不进入 Q(A)；无新增 state 节点。
- 根文件 SHA256：`A591B0135CFAB19247900968ECC7B4A3AFDDD7DB3BCDADB8EE93BC9EFA8225E7`
- 归档文件 SHA256：`A591B0135CFAB19247900968ECC7B4A3AFDDD7DB3BCDADB8EE93BC9EFA8225E7`

## 评测

```powershell
\.venv\Scripts\python.exe -m py_compile solution.py
\.venv\Scripts\python.exe -u evaluator\real_model_suite.py --models opt-125m --solution solution.py --candidate-name c59-headroom-blockwise-opt --device cpu --algorithm-device cuda --cache-mode read --seq 128 --calib 2 --test 4 --output artifacts\real_model_suite\c59-opt.json --report logs\evaluations\2026-08-29-c59-opt.md
```

| 模型 | Linear | Attention | Total | API |
| --- | ---: | ---: | ---: | ---: |
| OPT-125M | 35.639634 | 19.581565 | 55.221199 | 54.42s |

- OPT 从 v056 的 `69.889098` 回退 `−14.667899`；候选逐块自由度产生严重 calibration 过拟合，未继续运行其他模型。
- 官方得分/时间：`NA`；本地 official-flow 代理只用于相对排序。

## 结论

逐块产品贪心不能替代经过验证的整层 parent/headroom 裁判；v059 拒绝并恢复 v056 的整层选择。
