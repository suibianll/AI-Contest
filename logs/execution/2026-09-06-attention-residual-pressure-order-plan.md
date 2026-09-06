# Attention residual-pressure tied permutation 执行记录

日期：2026-09-06

计划：`2026-09-06-attention-residual-pressure-order-plan`

状态：**CLOSED / R1_REJECTED**

父版本：v189，SHA256 `261202248a0146a2ee45f3df60bd1979bb8171b7c162921013b0024c848617af`

候选源码：`workbench/attention_residual_pressure_permutation_solution.py`

候选 SHA256：`87375D16F2A71827243E633C4C464121E853393E63E91A2939EFC112B5285CAE`

## 执行

- R0：`py_compile`、六 API 独立导入、合成 GQA 调用、state/参数合法性、tied Q/K 连续
  QK 不变量和有限输出检查全部通过。
- R1 shard0：8 个 Attention case，delta mean `-0.005305`、L1 `0.020938`、正/负/零
  `2/2/4`；最坏 layer12/length128 为 `-0.061995`。
- R1 shard1：8 个 Attention case，delta mean `0`、L1 `0`、正/负/零 `0/0/8`。

R1 原始 eval-v3 结果：

`artifacts/proxy_v3/attention-residual-pressure-permutation-20260906/r1/`

结论：shard0 已触发负向与 L1 门禁，按预注册规则关闭。不运行 R2/OOD/default，不提交
官方，不扫描压力统计或阈值；根 `solution.py` 仍为 v186。
