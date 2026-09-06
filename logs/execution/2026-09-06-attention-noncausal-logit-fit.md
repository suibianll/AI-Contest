# Attention 无因果 logit-gain 拟合执行记录

日期：2026-09-06
计划：[`执行计划`](../../docs/superpowers/plans/2026-09-06-attention-noncausal-logit-fit-plan.md)
父：根 `solution.py` v189，官方 `17616/275s`

## 裁决

候选 SHA `65a22570b52bd8fd269934515b95b3d61b7b84518238696ef554728a7efcbe34` 通过 R0，
但 Attention 六 shard 配对的总体 delta 为 `-0.000302142974`，并在 shard 4 出现
`-0.002717013` 的系统性局部回归，正式 **REJECTED**。根 `solution.py` 不变；未创建
版本号、未生成官方提交包、未提交官网。

## R0：实现边界与合法性

- 以官方 v189 为唯一父，仅替换校准期 `_fit_attention_logit_gain` 的 causal 行中心
  统计为 full non-causal 行中心统计。
- 仍使用每 KV head 一个固定 gain、原 log shrink/clamp 和 D1 Q/K 折叠；连续 QK
  乘积保持不变，在线 API 数量和动态搜索不变。
- 候选单文件导入六个 API、`py_compile` 均通过；输出 finite，case identity 唯一。

## R1：六 shard 配对

固定 `eval-v3`、proxy-v2 cache、CUDA、Attention-only、父 v189：

| shard | mean | median | L1 | 正/负/零 |
|---:|---:|---:|---:|---:|
| 0 | `+0.000000994` | `-0.000388978` | `0.000677588` | `3/5/0` |
| 1 | `+0.000296864` | `+0.000922737` | `0.002091154` | `6/2/0` |
| 2 | `+0.000145829` | `+0.000490399` | `0.001143076` | `5/3/0` |
| 3 | `+0.000965521` | `-0.000316088` | `0.003268941` | `4/4/0` |
| 4 | `-0.002717013` | `-0.000085219` | `0.006592855` | `4/4/0` |
| 5 | `-0.000505053` | `-0.000521653` | `0.001481236` | `3/5/0` |

全体候选 mean `0.752470211867`，父 mean `0.752772354841`，delta
`-0.000302142974`。shard 4 的 layer 4 mean `-0.016109`、最坏单 case
`-0.030935`，且 worst-20% tail `-0.004729`；shard 5 仍为负。虽所有输出有限，
但该结果否定无因果拟合假设，未继续跑 OOD/default/time，未调整参数或扫描邻域。

## 证据

- R1：`artifacts/proxy_v3/attention-noncausal-logit-fit-20260906/r1-full/`
- R0 shard 0/1：`artifacts/proxy_v3/attention-noncausal-logit-fit-20260906/r1-shards01/`
- 源码：`workbench/attention_noncausal_logit_fit_solution.py`
