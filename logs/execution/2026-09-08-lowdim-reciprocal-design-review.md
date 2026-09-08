# 低维 A@W 与 Q/K 互逆设计筛选记录

日期：2026-09-08。性质：设计审查与计划登记；未运行模型评测，未修改根或候选源码。

## 保留

1. 官方标准误差归一化的最终 hard A@W 目标，统一写成 `MSE_player / MSE_STD`。
2. 固定低维基上的合法互逆 A/W 参数化 `A'=AT, W'=WT^-T`。
3. GQA group 共享的 Q/K 对角互逆参数化，V 冻结，独立 holdout 做真实 hard Attention output 选择。
4. sufficient statistics 仅用于固定 `X`、仿射 `W(theta)`、固定 code 区间内生成 proposal。

## 删除或降级

- `4400/5000 -> local gain 0.88` 没有证据，相关本地门槛删除。
- `sum_b g_b Z_b` 改变连续输出，不属于互逆等价变换。
- SSE 分子除以 MSE denominator 缺少 `numel`，计划统一改用 MSE/MSE。
- 两个独立模块分别优化后组合违反单一最高分根政策。
- 多输出组、多 basis、多 hierarchy、`2~5` 轮等范围构成隐性 sweep，不进入执行计划。
- 外部 `21071/283s` 的源码和调用图未绑定，只作方向证据和时间风险参考。

## 执行边界

当前唯一候选仍为当前根上的 LC3 objective-only。完整官方结果到账后，才允许从 D1/D2 中选一个
与届时根不等价的固定机制；所有候选从最新完整根构建，不产生侧父或组合阶段。
