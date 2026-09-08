# Attention 无因果 logit-gain 拟合执行计划

> 创建：2026-09-06  
> 状态：**CLOSED / R1_REJECTED**
> 父候选：v189 `static-actorder-hdiag-recovered`，根 v189 官方 `17616/275s`。

## 1. 假设

当前 Attention 主评分路径使用无因果 softmax，而 v186/v189 的固定 logit-gain
校准用因果三角区域的行中心 logits 拟合。若把同一固定 per-KV-head gain 改为在
无因果行中心 logits 上拟合，可能降低主评分路径的 Q/K 残差；该改动只发生在校准
阶段，仍以乘积保持不变的 Q/K multiplier 折叠进入普通状态，在线不增加搜索或 API。

这是一次单机制实验：不改变 center、permutation、block smooth、pair transform、
V state、动态 offset、refine budget、alpha 或候选数量。

## 2. 固定实现边界

- 输入、cache、device、evaluator 与 v189 相同；根 `solution.py` 不改，候选使用独立
  workbench 源码。
- 使用 `_ATTN_LOGIT_GAIN_TOKENS`、log shrink、有限夹断和 D1 折叠常量的现有固定值。
- 每个校准窗口取完整无因果 logits，按 query 行中心化；按 KV head 聚合 Q-head。
- 若统计无效，逐 head 回退 gain=1；不读取 holdout，不按 layer/模型路由。
- 仅保存合法 CPU 标量向量；不把 calibration Q/K/V 或输出张量写入 state。

## 3. 门禁与停止

1. R0：六 API 导入、合法 state、有限输出、Q/K 连续乘积不变量通过。
2. R1：Attention shard 0/1 配对；若出现系统性负向或 no-op，立即关闭，不扫窗口、
   shrink、alpha、factor 或候选邻域。
3. R2：若 R1 通过，完成 Attention 六 shard、OOD 与 fresh default；检查 L1<0.02、
   control、长度/层尾部及时间模型。分数只作同 cache 诊断，不能换算官方分数。
4. 只有 default Overall 严格高于当前本地最高 `0.687776303` 且预测官方时间 `<280s`
   时，才生成版本归档、`solution.zip`、提交并推送；否则关闭候选，根仍为 v189。

v189 官方回传已确认；不重复提交相同 SHA。新候选若满足门禁，才追加独立版本。

## 4. R1 裁决

候选 SHA `65a22570b52bd8fd269934515b95b3d61b7b84518238696ef554728a7efcbe34` 完成
Attention 六 shard、48 个配对 case；接口、有限输出、唯一 case identity 均通过。
六 shard 的 signed delta 如下：

| shard | mean | median | L1 | 正/负/零 |
|---:|---:|---:|---:|---:|
| 0 | `+0.000000994` | `-0.000388978` | `0.000677588` | `3/5/0` |
| 1 | `+0.000296864` | `+0.000922737` | `0.002091154` | `6/2/0` |
| 2 | `+0.000145829` | `+0.000490399` | `0.001143076` | `5/3/0` |
| 3 | `+0.000965521` | `-0.000316088` | `0.003268941` | `4/4/0` |
| 4 | `-0.002717013` | `-0.000085219` | `0.006592855` | `4/4/0` |
| 5 | `-0.000505053` | `-0.000521653` | `0.001481236` | `3/5/0` |

六 shard 合计候选 mean `0.752470211867`，父 v189 mean `0.752772354841`，delta
`-0.000302142974`。shard 4 的 layer 4 mean `-0.016109`、最坏 case `-0.030935`
以及负向 tail 触发停止条件；该机制 `REJECTED`，不运行 OOD/default/time，也不扫
window、shrink、alpha、factor 或候选邻域。根 `solution.py` 保持官方 v189。
