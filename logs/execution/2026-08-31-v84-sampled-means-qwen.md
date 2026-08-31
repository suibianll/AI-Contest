# Qwen 主模型本地评测报告

运行时间：2026-08-31 20:27:02（profile=sampled-means-v1，mode=amax6，seq=128，calib=2，test=4，cache_mode=read）

主评测配置：`qwen-official`，主模型 `qwen2.5-0.5b`；本地唯一主指标是抽样 Linear/Attention 的平均 gain。官方参考面板 250+200 仅作规则背景，不参与本地分数换算。
本地评测不复制 case、不拟合官方绝对分数。官方分数没有进入候选校准状态，也没有传给 `solution.py`。官方评测（2026-08-31 修订）不再限制任何 `A@W` 拟合用法，只限制端到端运行时间；候选可按需自由使用 `A@W` 优化 `Q(W)` 或 `Q(A)`。
官方上下文：外部 `youxilee/hif4` 用户提供结果为 24153/239s，仅作不可导入的参考；新增 2 个用例呈 Qwen 30B-like 特征，但完整输入尚未公开。

## 数据与模型完整性

- 数据集：`Salesforce/wikitext` / `wikitext-2-raw-v1` / revision `b08601e04326c79dfdd32d625aee71d232d685c3`。
- 评分协议：v5；标准 codec SHA256 `7fb21539c0556d10c77859e3e9ffb3d50dd3a2d0240e1cbb0923455e21bd6d3f`。
- calibration 来自 train，test 来自 validation；每个窗口来自一个文档，禁止环形重复、窗口重叠和跨 split 文档复用。
- Qwen2.5-0.5B（GQA、RoPE、SwiGLU）是默认主模型；显式加入的其他模型只作为独立 guardrail，缺失或轻微回退不会覆盖 Qwen 主分。
- 模型状态：

| 模型 | 状态 | 层数 | hidden | heads / kv-heads | 数据来源 | 说明 |
|---|---|---:|---:|---:|---|---|
| qwen2.5-0.5b | loaded | 24 | 896 | 14 / 2 | cache | qwen2 |

## 唯一主结果：组件平均得分

每个测试 case 的 gain 为 `(MSE_STD-MSE_PLAYER)/MSE_STD`；本表只报告所有抽样 Linear case 的算术平均和所有抽样 Attention case 的算术平均。数值是 0–1 比例，百分比列是同一数值乘 100。

| 模型 | 候选 | Linear mean | Linear mean (%) | Attention mean | Attention mean (%) | Cases (L/A) | Local API (s) | Wall (s) |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| qwen2.5-0.5b | v84-sampled | 0.477266 | 47.727 | 0.709020 | 70.902 | 224/32 | 422.615 | 433.624 |

抽样索引（所有候选共用，避免日志间样例漂移）：
- layers=`[0, 1, 5, 10, 13, 15, 22, 23]`; calibration_windows=`[0, 1]`; test_windows=`[0, 1, 2, 3]`; roles=`['q', 'k', 'v', 'o', 'fc_gate', 'fc_up', 'proj']`
- source cases=24 layers × 7 roles × 4 test windows；本次 cases=224 Linear + 32 Attention

## 抽样计划与使用边界

- profile：`sampled-means-v1`；`sampled-means-v1` 默认按层深分层抽取 8 层，保留全部 role，并按窗口位置抽取 4 个 validation window；calibration window 全部保留。
- seed：`20260831`；每个结果的 `sample_plan` 保存实际 layer/window index 和文档 ID，可逐字复现。
- 默认只评估 Qwen2.5-0.5B；需要跨模型 guardrail 时显式传入 `--models`，但各模型仍分别报告 mean，不相加。
- `official_flow_score`、`panel_score` 仅留在 JSON 兼容字段，不能作为当前主结果；本报告不展示它们。
- 本地时间只作为同一硬件、同一 cache、同一 shape 和同一抽样计划下的 A/B 指标。官方 300 秒是修订（2026-08-31）后的端到端运行限制，不能由本地 API 秒数直接判定。
- `cache_mode=read` 只读取已固定的模型前向快照；改变 seq/calib/test/layers 或数据 revision 必须先生成对应新 cache。
