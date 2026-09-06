# Attention residual-pressure tied permutation：REJECTED

- 日期：2026-09-06
- 状态：`CLOSED / R1_REJECTED`
- 父版本：v189，SHA256 `261202248a0146a2ee45f3df60bd1979bb8171b7c162921013b0024c848617af`
- 候选源码 SHA256：`87375D16F2A71827243E633C4C464121E853393E63E91A2939EFC112B5285CAE`
- 根正式版本：v186，未修改

## 结果

候选在父 `d/center` 坐标中统计实际 HiF4 基础重构残差，生成 tied Q/K 的 KV-head
permutation；不改变 scale、temperature、Fisher、Jacobian、4×4 或在线路径。R0 的
合法性、连续 QK 不变量和有限输出检查通过。

R1 使用固定 proxy-v2 cache、CUDA、v189 baseline 的 Attention eval-v3 前两片：

| shard | cases | delta mean | L1 | 正/负/零 | 最坏 delta |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 8 | -0.005305 | 0.020938 | 2/2/4 | -0.061995 |
| 1 | 8 | 0 | 0 | 0/0/8 | 0 |

shard0 已触发负向和 `L1>=0.02` 门禁，按计划停止；不运行 R2 六片、OOD 或 fresh
default，不提交官方，不扫描残差压力定义或阈值。

原始 eval-v3 证据：

- `artifacts/proxy_v3/attention-residual-pressure-permutation-20260906/r1/`
- `logs/execution/2026-09-06-attention-residual-pressure-order-plan.md`

官方分数/时间：`unregistered/NA`。本地 proxy 不换算官方分数。
