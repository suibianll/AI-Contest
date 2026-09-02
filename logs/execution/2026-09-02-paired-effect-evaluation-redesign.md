# 配对机制评测改造记录

日期：2026-09-02

## 问题

旧的 `--linear-cases 14/56` 是默认序列前缀，分别只覆盖模型前 2/8 层，不是纵深采样；父候选
比较也主要依赖 aggregate mean 手工相减。因此小幅变化无法区分稳定改善、层间抵消、目标 role
内部抵消、未修改路径泄漏和 W/A 来源。

## 实现

`evaluator/official_eval.py` 新增：

- `--effect-panel`：8 个纵深层 × 全 7 Linear role，共 56 cases；另取 5 个覆盖深度和公开长度
  的 Attention 哨兵。校准仍建立完整 168 Weight + 24 Attention state。
- `--baseline-json`：候选运行后与保存父版本按
  `(layer, role, window, split, length)` 精确配对。
- `--candidate-json`：对两个已有 JSON 零 API 重放。
- `--focus-linear-roles`：支持具体 role 或 family，例如 `fc` 自动匹配 `fc_gate/fc_up`，其余
  case 作为 control。
- `paired_effect`：报告 overall/focus/control，mean/median signed delta，改善/回归/不变数，
  MSE ratio，逐 role/family/layer/shape/split/length，W/A 与 Q/K/V 控制臂差分，最好/最坏 case
  和同机 API 时间差。
- 严格配对校验：case identity、`mse_standard`、`reference_energy` 不同即拒绝比较。

效果标签只按逐 case 符号描述为 `no_effect`、`consistent_improvement`、
`consistent_regression` 或 `mixed`，不设置新的分数阈值。

## 历史结果重放

v152：

```powershell
.venv\Scripts\python.exe evaluator\official_eval.py `
  --baseline-json artifacts\official_eval\v152-parent-56.json `
  --candidate-json artifacts\official_eval\v152-fc-cat-off-56.json `
  --focus-linear-roles fc `
  --output artifacts\official_eval\v152-fc-cat-off-paired-effect.json `
  --report logs\official_eval\v152-fc-cat-off-paired-effect.md
```

- Linear overall：mean Δ `+0.000186`，3 改善 / 3 回归 / 50 不变，`mixed`。
- focus fc：mean Δ `+0.000653`，3/3/10，`mixed`。
- fc_gate `+0.001871`，fc_up `−0.000565`。
- control 40 cases 与 Attention 均 `no_effect`。
- 最坏层：layer 7 fc_gate `−0.007283`；layer 0/1 fc_up
  `−0.003087/−0.001435`。

结论：原先的 `+0.000187` 是 role/层间正负抵消，不是稳定改善。

v153：

```powershell
.venv\Scripts\python.exe evaluator\official_eval.py `
  --baseline-json artifacts\official_eval\v151-parent-targeted.json `
  --candidate-json artifacts\official_eval\v153-fc-decoupled-activation-targeted.json `
  --focus-linear-roles fc `
  --output artifacts\official_eval\v153-fc-decoupled-paired-effect.json `
  --report logs\official_eval\v153-fc-decoupled-paired-effect.md
```

- focus fc：mean Δ `−0.048211`，median `−0.049511`，0 改善 / 4 回归 / 0 不变，
  median player-MSE ratio `1.078387`，`consistent_regression`。
- 10 个 control 与 Attention 均 `no_effect`。

结论：该机制是纯 fc 回归，不是全局噪声或路由污染。

## 后续固定流程

1. 每个父版本只生成一次 effect baseline JSON。
2. 候选声明实际修改的 focus role，运行同一 effect panel 并配对。
3. 先判 focus 的方向与符号一致性，再查 control、误差来源和最坏 case。
4. 有一致、可解释的机制信号后，才跑默认 168 Linear + 120 Attention panel。
5. `14/56` prefix 只用于接口/格式 smoke；full Cartesian 只用于 stress。

## 验证

- `python -m py_compile evaluator/official_eval.py`
- `python -m pytest -q`
- 两个历史 JSON paired replay 均成功生成 JSON 与 Markdown 报告。
