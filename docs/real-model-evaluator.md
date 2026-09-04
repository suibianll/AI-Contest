# 旧评测器退役说明

本文档保留作迁移索引。`evaluator/real_model_suite.py`、`real_data_eval.py`、
`holdout_eval.py` 和 `synthetic_attention_eval.py` 属于旧的 sampled evaluator，已不再是
活动工具；它们不能代表官方变长 Attention 校准，也不能用于时间或分数排序。

当前日常入口是 [`evaluator/eval.py`](../evaluator/eval.py)，协议为 `eval-v3`；它底层读取
[`evaluator/official_eval.py`](../evaluator/official_eval.py) 生成的 `proxy-v2` cache，使用
共享校准调用图（168 Weight + 24 Attention state）、默认 168 Linear +
120 Attention 分层真实 W/A panel，用 Attention 校准长度
`[10,128,512,1024,1024]`，执行独立参数/state 校验，并输出 `linear_mean`、
`attention_mean`、`overall_mean`、六 API 的 `api_total_seconds` 和 `wall_seconds`。
旧 `official-shape-v1` 已退役，其 JSON 隔离在 `artifacts/official_eval/legacy-v1/`。

批量复评命令（eval-v3 official audit）：

```powershell
.venv\Scripts\python.exe -u evaluator\eval.py --official-audit `
  --cohort new-weight --scenario both --shards 0,1,2,3,4,5 `
  --cache artifacts\official_eval\cache\qwen2.5-0.5b-proxy-v2.pt `
  --calibration-cache-mode auto --algorithm-device cuda `
  --output-dir artifacts\proxy_v3\official-audit
```

历史 sampled JSON 已移入带日期的 `artifacts/archive/` recovery 目录，只能作为旧实验取证，
不能与新协议结果混排；新实验不得写回 `artifacts/real_model_suite/`。
