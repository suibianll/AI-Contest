# C66 动态激活损失覆盖目标 1.0（本地接受）

- 日期：2026-08-29
- 版本：v066 / C66
- 父版本：v063 / C63
- 唯一机制：将数据驱动的动态激活困难块损失覆盖目标由 `0.999` 提高到 `1.0`，允许覆盖完整损失尾部；仍受 `max_refine_blocks` 上限和现有合法性检查约束。Linear 候选、静态 A@W Q(W)、CAT 与 Attention 选择逻辑不变。
- 根文件 SHA256：`F37084D0DFF548D9C6A8D57D87C77B0CFEEB4C6976E95A24F797427C32A16B26`
- 归档文件 SHA256：`F37084D0DFF548D9C6A8D57D87C77B0CFEEB4C6976E95A24F797427C32A16B26`

## 验证

```powershell
\.venv\Scripts\python.exe -m py_compile solution.py
\.venv\Scripts\python.exe -u evaluator\real_model_suite.py --models gpt2-small opt-125m qwen2.5-0.5b --solution solution.py --candidate-name c66-activation-ratio100-screen --device cpu --algorithm-device cuda --cache-mode read --seq 128 --calib 2 --test 4 --output artifacts\real_model_suite\c66-screen.json --report logs\evaluations\2026-08-29-c66-screen.md
\.venv\Scripts\python.exe -u evaluator\real_model_suite.py --models gpt2-medium pythia-160m --solution solution.py --candidate-name c66-activation-ratio100-rest --device cpu --algorithm-device cuda --cache-mode read --seq 128 --calib 2 --test 4 --output artifacts\real_model_suite\c66-rest.json --report logs\evaluations\2026-08-29-c66-rest.md
```

| 模型 | Linear | Attention | Total | API |
| --- | ---: | ---: | ---: | ---: |
| GPT-2 small | 134.327340 | 21.306236 | 155.633576 | 55.78s |
| GPT-2 medium | 231.098280 | 43.760024 | 274.858304 | 137.58s |
| OPT-125M | 65.472279 | 19.647602 | 85.119881 | 54.29s |
| Pythia-160M | 138.290983 | 40.647879 | 178.938863 | 55.24s |
| Qwen2.5-0.5B | 287.032704 | 63.119717 | 350.152420 | 151.91s |
| **五模型合计** |  |  | **1044.703044** |  |

- 相对 C63 合计 `1044.096647` 提升 `+0.606397`；五模型均正向（GPT‑2 small `+0.196592`、medium `+0.010068`、OPT `+0.075238`、Pythia `+0.041965`、Qwen `+0.282534`）。
- API 时间均低于 300s；状态与接口格式未改变。
- 官方得分/时间：`NA`；本地 official-flow 代理只用于相对排序。

## 结论

全损失尾部覆盖带来稳定的小幅收益，且未触发时间上限。C66 作为当前本地父版本保留。
