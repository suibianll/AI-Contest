# C71 proj H32/H64 + 最终量化器候选排序（本地拒绝）

- 日期：2026-08-29
- 版本：v071 / C71
- 父版本：C69
- 唯一机制：下投影（`out_features < in_features`）扩展 block-S 到
  `4/8/16/32/64`，并在候选排名中使用与部署一致的 bounded offset/refine
  最终量化器。
- 目的：隔离复现外部 v2.1/v2.5 的 proj H32/H64 与 objective-match 思路；
  未加入 C70 联合残差补偿。
- 根文件/归档文件 SHA256：
  `DA7837626368A4EB0A29C8B66B682BBA84B4C1323A653866DE7CA28656C2C1CC`

## 验证

```powershell
\.venv\Scripts\python.exe -m py_compile solution.py
\.venv\Scripts\python.exe -u evaluator\real_model_suite.py --models gpt2-small opt-125m qwen2.5-0.5b --solution solution.py --candidate-name c71-proj-final-quantizer-screen --device cpu --algorithm-device cuda --cache-mode read --seq 128 --calib 2 --test 4 --output artifacts\real_model_suite\c71-proj-final-quantizer-screen.json --report logs\evaluations\2026-08-29-c71-proj-final-quantizer-screen.md
```

| 模型 | Linear | Attention | Total | API |
| --- | ---: | ---: | ---: | ---: |
| GPT-2 small | 142.657544 | 21.306236 | **163.963780** | 64.72s |
| OPT-125M | −73.851750 | 19.647602 | **−54.204148** | 63.50s |
| Qwen2.5-0.5B | 317.769616 | 63.119717 | **380.889333** | 188.98s |

相对 C69：GPT-2 small `+8.327712`，OPT `−139.324449`，Qwen `+30.736913`。
OPT 的负向回退违反可迁移性，C71 不晋级；根恢复 C69。

## 诊断结论

只增加 H32/H64、仍使用旧 operand-local 代理时，三模型逐位不变；真正改变
结果的是最终量化器排名，但它在当前 CAT/FULL64/产品选择父版本上选择了
OPT 的灾难性候选。外部 changelog 的 proj `+0.008` 不能脱离其 v2.6
父版本、候选族和 calibration 分布直接搬运。后续若再用 H32/H64，必须把
候选限制为低自由度静态 Q(W) 池并做跨 fold 软选择，不能全量替换部署量化器。
