# C66 动态激活损失覆盖目标 1.0（官方确认）

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
- 上表 API 时间是旧本地 official-flow 代理的分模型记录，只用于相对排序；状态与接口格式未改变。

## 修订版官方评测结果（用户确认）

- 评测面板：250 个 Linear case + 200 个 Attention case。
- 官方时间限制：`420s`（7 分钟）。
- 官方得分/时间：**`22557 / 217.2s`**。
- 相对新版本地归档锚点：v031/C39-FW `21864 / 161.3s`、v034/C41b
  `21864 / 159.4s`、v051/C47b `22451 / 234s`；C66 较 v051 提升 `106` 分，
  同时快 `16.8s`。
- 当前本地归档官方冠军仍低于外部 [`youxilee/hif4`](https://github.com/youxilee/hif4)
  的用户提供结果 `24153 / 239s`：分数差 `1596`，时间差 `21.8s`。外部代码
  未导入本仓库，以上数字不视为本地复现。

## 结论

全损失尾部覆盖带来稳定的小幅收益，且未触发时间上限。C66 已由官方新版面板
确认，并作为当前本地归档官方冠军；后续实验应以 C66/C69 的合法校准路径为父版本，
优先补齐外部实现中的 Linear 输出级联合残差补偿。
