# Qwen 主面板本地评测器重构

日期：2026-08-29  
状态：已实现并提交  
协议：`real_model_suite.py` scoring protocol v3

## 背景

新版官方面板为 250 个 Linear case + 200 个 Attention case，时间限制为
420 秒。当前本地可用的真实快照来自 Qwen2.5-0.5B（GQA、RoPE、SwiGLU）等
模型，原评测器按模型层数、角色数和本地窗口数直接累加，五模型 raw sum 会
改变同一候选的权重。官方新增的两个用例只有 Qwen 30B-like 的统计特征，完整
输入尚未公开；外部 `youxilee/hif4` 的 `24153 / 239s` 只作不可导入参考。

## 设计

- 默认 `--panel-profile qwen-official`，主模型为
  `qwen2.5-0.5b`，目标面板固定为 250 Linear + 200 Attention。
- 每个组件先按冻结语料的 native case score 取均值，再计算
  `panel_score.total = 250 * Linear_mean + 200 * Attention_mean`。
- 不复制本地 case、不伪造隐藏样例、不将 panel 分数换算成 Official Score；
  JSON 同时保留 native `official_flow_score` 便于旧报告回溯。
- 其他模型仍可同时评测，但只作为 soft guardrail；缺失或轻微回退不会覆盖
  Qwen 主排序。主模型仍执行合法 state、非 finite 和 `<420s` 检查，没有新增
  增益、coverage 或组件正向门槛。
- 当前修订官方锚点登记为 C39 `21864/161.3s`、C41b `21864/159.4s`、
  C47b `22451/234s`、C66 `22557/217.2s`；旧面板锚点保留历史字段但不参加
  当前排序审计。

## 实现内容

- `evaluator/real_model_suite.py`
  - 新增固定面板常量、外部参考上下文和当前锚点元数据；
  - 新增 `build_panel_score()`，为旧 JSON 提供兼容回退；
  - 新增 `--panel-profile`、`--primary-model`；
  - audit 输出 `primary_panel_score_*`、guardrail 均值、主排序和软有效性；
  - 默认报告改为 Qwen 主面板，同时保留 native raw order。
- `tests/test_real_model_suite.py` 新增面板均值守恒、Qwen 主模型/guardrail、
  当前官方锚点清单测试。
- README、英文 README、评测器说明和 22000+ 优化计划同步更新。

## 验证

- `python -m py_compile evaluator/real_model_suite.py tests/test_real_model_suite.py`：通过。
- 面板/锚点/排序相关测试：`6 passed`。
- 全测试（使用工作区 basetemp）：`80 passed, 2 skipped`；剩余 4 个失败为
  既有环境或 C69 期望不一致：缺少 `transformers` 的两个测试，以及旧的
  `_ACTIVATION_QUADRATIC8_MAX_RATIO==0.08`、C21-C narrow state 断言。
- Qwen 真实缓存 1 层/1 窗口链路：native `7/1` source cases，panel `250/200`，
  `panel_score.total=314.895424`，API `21.250s`。该缩小配置仅验证字段和流程。

完整模型缓存、模型权重、评测 JSON 和 Markdown 报告均留在本地忽略目录，未
加入 Git；三个用户未跟踪文档也未修改。
