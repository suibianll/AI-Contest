# LC1 / LC2 方法审计与下一步纠偏

日期：2026-09-08。性质：零 API 代码与证据审计；未运行模型评测，未修改候选或根 `solution.py`。

## 结论

1. L-C1 机制卡规定只把根的 fold 权重从 `1/||Y_f||^2` 替换为
   `1/(F*numel*MSE_STD_f)`，并明确不改 activation、Attention 或其他机制。
2. `continuous_linear_lc1-root-aw-refine_rejected` 实际实现为 rank-8 residual-subspace A@W
   refinement。它不是原卡的 objective-only 修改。因此 `delta=-0.302577` 只拒绝该 rank-8 根后处理，
   原 L-C1 问题仍未执行。
3. LC2 文档声称逐元素尝试 mantissa `±1` 和 zero sign flip；实际循环对整个 64-block 的
   `mant_cur` 张量统一加或减 `0.25`，只形成两个整块同步提案。`sign_b=sign(Wb)` 使零值保持
   `sign=0`，所谓 zero sign flip 没有发生。
4. 因而 LC2 的 `accepted=0` 不能证明不存在逐元素合法邻居，也不能证明根在 A@W 目标上局部最优
   或已经饱和。历史 manifest 保留原文作审计链，当前解释由本记录取代。

## 方法根因

- 机制卡没有逐项绑定实际 diff，导致实现者在原卡不可直接落地时替换了求解器，却继续沿用原卡因果解释。
- 搜索粒度和边界行为没有合成测试，导致“逐元素”“zero sign flip”等文字与实际张量操作不一致。
- 单一失败实现被扩写为坐标或机制族的饱和结论，超过实验支持范围。

## 修订后的下一步

1. R0 零 API 审计根是否存在 objective-only fold 加权入口；没有则记 `NOT_APPLICABLE`，不替换机制。
2. 入口存在时注册 L-C3，只改权重表达式；合成测试先证明 diff 的单义性，再做 shard0 和官方裁决。
3. 随后对 A23 官方正向 Q/K 互逆 scale 实现做最小差异审计，再移植到完整根；不同时更换训练目标、
   求解器或主干。
4. 正式候选继续从完整根构建；侧隔离只用于因果诊断，不形成并行提交父。
