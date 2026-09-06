# 2026-09-06 Linear 静态 activation-GPTQ 条件曲率执行记录

## 结论

**CLOSED / C1_REJECTED**。候选将 v189 静态 activation-GPTQ 的完整 64-channel
block 排序从部署权重列能量 `diag(H)` 改为已存 activation-GPTQ inverse Gram 的
`sum(1 / diag(H^-1))`。排序真实生效，但两个首 shard 都负向，按预注册规则关闭。

## C0

- 候选：`workbench/linear_static_gptq_conditional_curvature_solution.py`
- SHA256：`54638e1146a07558b019c098f1496b3fa5d33f6805026e08a5ff855338e1d5ec`
- `py_compile`、单文件导入、条件曲率有限性与合成排序 smoke：PASS。
- 运行日志出现 `[STATIC-ACTORDER-CONDITIONAL-CURVATURE] reachable=1`；state 中块序
  已生成，非回退路径。
- 根 `solution.py` SHA256 仍为
  `F8495DCA20334ACBDAD16FC18EE41A4970F31E1837FDEEDCEE9C70AEE54E7EB8`。

## C1

固定 v189 baseline、`evaluator/eval.py` eval-v3、proxy-v2 cache、CUDA、Linear shard
`0,1,2,3,4,5` 配对运行。评测因负向门禁在 shard 1 后停止，实际覆盖 112 个 case；未
运行 default/OOD、跨模型或官方提交。

| shard | delta mean | delta median | L1 | 正/负/零 | 主要回归 |
|---:|---:|---:|---:|---:|---|
| 0 | -0.000491 | -0.000519 | 0.001316 | 18/38/0 | `o` mean -0.001068 |
| 1 | -0.000125 | -0.000176 | 0.001128 | 27/29/0 | `fc_gate` mean -0.001287 |

说明：shard 0 的完整 median 在 evaluator 的 JSON 中保留；其 delta mean 已足以触发
计划门禁。候选整体 mean `0.6342678291013429`，C1 不具有 default-panel 或官方分数
等价性。

## 归档与后续

源码、manifest、结果已归档到
`solutions/20260906_linear-static-gptq-conditional-curvature_rejected/`；证据目录为
`artifacts/proxy_v3/linear-static-gptq-conditional-curvature-20260906/c1/`。

该单一块序规则不扫描 `diag(H)`/`diag(H^-1)` 混合权重、反向排序、ratio、seed 或
layer/role 邻域。下一活动计划必须登记新的数学机制。
