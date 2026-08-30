# v105 L1 full-hierarchy Weight-LRH（rejected）

## 裁决

该候选完成了 L1 的合成测试与 Qwen 五层×七 role 的 cross-fold 分层预筛，
但没有通过部署 parent 门禁，因此不替换根目录版本，也不进行 24 层 full-layer
评测。预筛 `both_player=0.523019429222563`，与 L0 基线逐条相同。

## 实现范围

- 跨 block rank-8 LRH，最多选择 4 个输入 block；
- 每个候选原子搜索 E6M2 scale 的合法局部 offset、lv2、lv3 和 15 个 signed
  mantissa level；
- 新 mantissa 使用新 hierarchy denominator 解码；
- 每个 fold 独立生成，交换 fold 做 cross-fold admission；
- 失败时返回 parent。

## 预筛统计

- cases：35（层 `{0,5,11,17,23}` × 7 roles）；
- LRH candidates：70（每个 case 两个 calibration fold）；
- cross-fold admitted：1/70；
- 相对 v100 stable parent 的最终字段改动：0/35；
- 分层 screen elapsed：265.871 s；
- L1 diagnostic elapsed：175.931 s；
- solution LF SHA256：`b5c9abf4738cdcab9ff14b34881795dc9ff0297622804f2d9dccc13fe0e7d004`。

唯一 cross-fold admitted 的候选位于 layer 23 / `proj`，但最终稳健 selector 仍
选择已有 HSDQ candidate，故输出与 stable parent 完全一致。LRH 在单 fold 上的
大幅下降伴随另一 fold 的恶化，说明当前两折数据下跨 fold 泛化不足。

## 证据

- [`l1-lrh-stratified-qwen.json`](../../../artifacts/real_model_suite/l1-lrh-stratified-qwen.json)
- [`2026-08-30-l1-lrh-stratified.md`](../../../logs/execution/2026-08-30-l1-lrh-stratified.md)
- [`2026-08-30-l1-full-hierarchy-lrh.md`](../../../logs/execution/2026-08-30-l1-full-hierarchy-lrh.md)

## 后续

按 active plan 的失败处理进入 L2；不得继续扩大该 LRH 的 rank、block 数或
sweep。此目录保留完整候选源码和测试，仅用于复现与审计。
