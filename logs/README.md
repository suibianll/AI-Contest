# 运行产物归档

这里存放运行报告，机器可读结果存放在 `artifacts/official_eval/`。源码归档在
`solutions/`，设计文档在 `docs/`；不要把三类内容混写。

## 当前唯一评测入口

`evaluator/official_eval.py` 使用 `official-shape-v1`：Qwen2.5-0.5B、250 Linear +
200 Attention、Attention calibration `[10,128,512,1024,1024]`、独立 HiF4 codec 和六个
官方 API。报告的主指标是 `linear_mean`、`attention_mean`，计时同时记录
`api_total_seconds` 与 `wall_seconds`。本地秒数不能转换为鲲鹏 920B 的官方秒数。

```powershell
.venv\Scripts\python.exe -u evaluator\official_eval.py --archive `
  --cache artifacts\official_eval\cache\qwen2.5-0.5b-official-shape-v1.pt `
  --cache-mode read --algorithm-device cuda `
  --output artifacts\official_eval\archive-official-shape-v1.json `
  --report logs\official_eval\archive-official-shape-v1.md
```

`logs/official_eval/` 是活动报告目录。旧 `logs/evaluations/`、`artifacts/real_model_suite/`
和 `sampled-means-v1/v2` 已移出活动路径到带日期的 `logs/archive/`、`artifacts/archive/`，
禁止再用于排序、调参或时间判定。

## 结果命名与留存

- 每次运行显式指定唯一 JSON 和 Markdown 路径，禁止覆盖另一次运行。
- 报告必须写协议、模型/数据 revision、校准长度、case 数、设备、API 调用次数、两项均值、
  API/Wall 时间和源 SHA256。
- 官方回传只追加到对应候选的历史记录；本地代理值不能填入 Official score/time。
- 清理旧结果时保留 `solutions/` 源码和官方回传日志；旧评测器产生的 JSON/MD 已移入历史
  归档，大缓存不提交 Git。活动目录只允许保留 canonical JSON/Markdown。
