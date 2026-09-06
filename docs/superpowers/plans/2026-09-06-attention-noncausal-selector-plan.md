# Attention mask-aligned output selector 执行计划

> 创建：2026-09-06
> 状态：**CLOSED / R2_REJECTED**
> 父版本：v189 `static-actorder-hdiag-recovered`，官方 `17616/275s`

## 1. 假设

`eval-v3` 的 Attention 主分数使用 non-causal softmax，而当前 A1 校准选择器把
causal output MSE 作为 primary、non-causal 作为 safety。将同一组已存在的 A1 候选
与终验门对调主次，使 non-causal 成为 primary、causal 成为 safety，可能让校准选择
更贴近实际评分目标。

这是一次单机制目标对齐实验：不改变候选生成、center、permutation、block smooth、
pair transform、V state、logit-gain、动态 offset、refine budget、alpha 或候选数量。

## 2. 固定实现边界

- 输入、cache、device、evaluator 与 v189 相同；根 `solution.py` 不改，候选只在
  workbench 中实现。
- 仅交换 A1 输出选择器和 `_a1_gate_passes` 的 causal/non-causal 角色；proxy 轨、
  状态格式、连续 QK 不变量和在线六 API 保持不变。
- 仍保留两条 mask 的逐 case safety 检查；不读取 holdout，不按 layer/model 路由。

## 3. 门禁与停止

1. R0：六 API 导入、合法 state、有限输出、Q·K 连续乘积不变量通过。
2. R1：Attention shard 0/1 配对；若系统性负向或逐位无变化，立即关闭，不扫任何
   参数邻域。
3. R2：若 R1 通过，完成 Attention 六 shard、OOD 和 fresh default；检查 L1<0.02、
   control、长度/层尾部及时间模型。
4. 只有 default Overall 严格高于当前本地最高 `0.687776303` 且预测官方时间 `<280s`
   时，才生成版本归档、`solution.zip`、提交并推送；否则关闭候选，根仍为 v189。

官方 v189 已回传；不重复提交 v189 相同 SHA。

## 4. 执行裁决（2026-09-06）

- R0 通过：候选六 API 可导入，`py_compile` 通过；候选 SHA256 为
  `de1e07c2515062298a575f5e2f6a1748a60bfab948fae67f4b27d32bc97fd1fb`。
- R1 Attention eval-v3 前两片均为正：shard0 delta mean `+0.013737235`、L1
  `0.013737235`，shard1 `+0.013085379`、L1 `0.013085379`；输出有限、case 唯一、
  coverage 通过。
- R2 eval-v3 六片完整结果为 `+0.001984898562`（candidate Attention mean
  `0.754757253403`，父 `0.752772354841`）。但该分片面板仅含 48 个 Attention case，
  不能替代 default-panel。OOD 的 `Δ(gain_in-gain_ood)=+0.002404319228`，在
  `|Δgap|<=0.01` 内；输出有限、case 唯一、coverage 通过。
- 兼容后端 fresh default（168 Linear + 120 Attention）才是本计划的 default gate：
  candidate Linear `0.640258324430`、Attention `0.748924596433`、Overall
  `0.685535937765`，相对 v189 Overall `-0.001353671077956`，相对本地最高
  `0.687776303` 为 `-0.002240365235489`。因此分数门未通过。
- 同一 fresh default 的 API 分解为 `W_calib=274.630146s`、`A_calib=60.748621s`、
  `dyn_act=60.692522s`、`dyn_qkv=3.064692s`，时间模型预测 `283.748107s`，也未通过
  `<280s` 门。候选不归档为正式版本、不生成提交包、不提交官方；根
  `solution.py` 保持 v189。

详细证据见
[`mask selector 执行记录`](../../../logs/execution/2026-09-06-attention-noncausal-selector.md)。
