# eval-v3 评估系统

日常流程以 [4B 测试指引](4b-panel-testing-guide.md) 为准，命令统一走 `evaluator/eval.py`。
默认 cache 为 `artifacts/official_eval/cache/qwen3.5-4b-proxy-v2.pt`，缺失时报错，不自动捕获 0.5B。

## 面板与隔离

六 shard 合计 Linear 336 case、Attention 72 case。4B 为异构结构，Attention 只覆盖六个
full-attention 层；不能用旧“每 shard 8 个 Attention”推断当前覆盖，实际以 manifest 为准。
校准与 validation/test 独立，评价最终 Q(A)Q(W)^T 或 Attention 输出。

父子必须匹配 evaluator、协议、cache/panel、设备、源码 SHA 与 case 身份及标准参考误差。
`--reuse-existing` 检查源码路径/SHA、缓存路径、场景与分片；不匹配会重新执行。
它不完整校验同路径缓存内容、evaluator 版本和设备；仅在这些均未变化且有原 manifest 证明时复用。
不能把该参数本身当作完整身份保证。
校准产物按源码与输入身份缓存；命中会标注 calibration_cache_hit，API 时间只记录。

## 分析与裁决

`evaluator/proxy_v3_analyze.py --baseline <parent.json> --candidate <candidate.json>` 提供配对
均值、中位数、总 L1、尾部、分组、control 和 API 热点。这些统计只用于合法性、可达性与失败诊断，
不再构成候选排序、提交或晋级门；正式裁决只看官方总分与官方 300s。

本地时间预测和 280s 门已移除；旧 JSON 字段 predicted_official_seconds/under_280_gate
为兼容读取保留为 null。分析器通过不代表官方晋级或完整提交检查完成。
官方分数/时间及同源码 SHA 才能确认晋级，官方时间硬限 300s。

## 退役和保留

- 不新增 0.5B、逐候选 OOD、跨模型 GPT-2/opt、fresh-default 计时测试。
- 日常入口拒绝 `--ood`：当前 4B capture 没有 OOD 输入，不能退回旧 OOD cache。
- `official_eval.py`、`proxy_v3_eval.py` 及参考合法性代码仍是运行依赖，保留为后端。
- `--official-audit` 是显式历史审计功能，不是候选日常流程，不自动批量重评。
- 历史 JSON/report、归档源码和回归测试保留，不与当前 4B 排名混用。

审计发现、修改范围和验证见[评估系统审计](evaluation-system-audit-2026-09-07.md)。
