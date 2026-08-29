# C68 A@W 静态块预算 15%（归档，拒绝）

- 日期：2026-08-29
- 版本：v068 / C68
- 父版本：v066 / C66
- 唯一机制：将静态 A@W Q(W) 条件更新选择的块预算由 `12.5%` 调到 `15%`；其余 Linear 512 行统计、CAT、headroom 和在线路径不变。
- 根文件 SHA256：`5F873E9424EA5040844085D97A738C5DAADF0AF72554C8BCAC0F710A63B397FC`
- 归档文件 SHA256：`5F873E9424EA5040844085D97A738C5DAADF0AF72554C8BCAC0F710A63B397FC`

## 验证

```powershell
\.venv\Scripts\python.exe -m py_compile solution.py
\.venv\Scripts\python.exe -u evaluator\real_model_suite.py --models gpt2-small opt-125m qwen2.5-0.5b --solution solution.py --candidate-name c68-product-ratio15-screen --device cpu --algorithm-device cuda --cache-mode read --seq 128 --calib 2 --test 4 --output artifacts\real_model_suite\c68-screen.json --report logs\evaluations\2026-08-29-c68-screen.md
```

| 模型 | Linear | Attention | Total | API |
| --- | ---: | ---: | ---: | ---: |
| GPT-2 small | 134.416424 | 21.306236 | 155.722660 | 57.98s |
| OPT-125M | 65.246115 | 19.647602 | 84.893717 | 59.40s |
| Qwen2.5-0.5B | 286.922986 | 63.119717 | 350.042703 | 159.93s |

- GPT‑2 small 较 C66 `+0.089084`，但 OPT `−0.226164`、Qwen `−0.109717`；三模型小计较 C66 回退 `−0.246797`。
- 官方得分/时间：`NA`；本地 official-flow 代理只用于相对排序。

## 结论

15% 预算已开始引入跨模型过拟合，保留 C66 的 12.5% 块选择预算。
