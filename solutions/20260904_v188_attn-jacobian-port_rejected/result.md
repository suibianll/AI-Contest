# v188 Attention Jacobian 敏感度移植（v186 + v187 机制）

> 状态：OFFICIAL REJECTED — `17595 / 268s`
>
> Parent：v186（官方 17599 / 272s，SHA `F8495DCA...7EB8`）。唯一改动：把 v187 的
> attention 输出 Jacobian 一阶敏感度 importance 作为校准最终步移植进 v186，
> LOO 门控通过后替换 Q/K state importance；在线路径零新增计算。

## 机制

在 logit gain 之后、返回 state 之前：对最终变换坐标系（multiplier/permutation/
center/rotation/block-smooth/pair\_transform 全链）上的校准 Q/K 计算 attention
输出一阶 Jacobian 敏感度（Q 侧含跨 key softmax 耦合项），压缩为 KV-group 共享的
KV-head×64 importance：逐 fold causal/non-causal 0.5 平均、跨 fold median、
log-space 1/4 收缩、\[0.5, 2.0] clamp；leave-one-fold-out 部署 MSE 门
（median 相对增益 > 0.0005 且 worst > −0.01，v187 预注册常量，无邻域扫描）。
V 与 Linear 完全不变。

## 本地结果

- 六 API smoke、py\_compile、脱离仓库单文件导入：通过；

- compact 4 哨兵层：no\_effect（0/0/4，门控全拒），校准 API +1.18s；

- default 120（vs v186）：Δmean **+0.000426**、L1 **0.001114**，6+/4−/110=；

- 门控 2/24 层通过（L12、L22——恰为无 pair\_transform 的层；约 18 层的 importance
  槽位已被 pair-smooth 输出拟合 importance 占据，Jacobian 无法胜出；静态
  importance 机制族在 v186 上实测饱和）；

- 变化明细：128/512/1024 净全正（合计 +0.092），len-10 净负（−0.037）；

- OOD（vs v186）：Δgap = +0.000426（|Δgap| ≤ 0.01 门内通过；OOD profile 校准下
  门控 0/24 全拒，OOD 输出与父逐位一致，无过拟合信号）；

- gpt2 跨模型（记录项，不作门禁）：Δmean −0.000950（4+/6−/50=）；

- 时间：Qwen 校准 API +3.14s 本地（gpt2 +1.885s/12 层一致），时间模型预测官方
  272 + 0.694×3.14 ≈ **274s**（< 280 提交门内）。

源码 SHA256：`1D14B4657D37CC017FD7EFADA9DC52BFA3A3783B5915A71F45FB5FBEAEBD67D2`

## 预注册裁决标准

- RETAINED：score > 17599 且 time < 300s（成为新完整官方父，根目录同步）；

- REJECTED：score ≤ 17599 且 time < 300s（根目录回滚 v186，Jacobian 移植族关闭）；

- TIMEOUT：time > 300s（根目录回滚 v186，按规则不缩窗重试）。

## 官方

- Score：**17595**（step_gain −4 vs v186 17599）
- Time：**268s**（−4s vs v186 272s；时间模型预测 274s，MAE 带内）

## 裁决与教训

按预注册规则 REJECTED：根 `solution.py` 已回滚 v186，Jacobian 移植族关闭
（不扫收缩/clamp/gate 邻域）。官方提交次数没有限制；历史配额解释已作废。

1. **符号门禁首次失手**：本地 Δmean +0.000426 / L1 0.001114 满足
   `Δmean>0 且 L1<0.02`，官方却 −4。该门禁历史 5/5 零误的记录被打破——
   但失手形态是「近零信号」：本地改动极小（110/120 case 不变）时门禁无判别力，
   官方 ±1~4 属于有效测量噪声带（v182→v186 三次单点增益也是 +1/+1/+3）。
   修正理解：门禁拦截的是大损失（−165~−1164 级），**不保证近零信号非负**。
2. **时间模型继续有效**：预测 274s vs 实测 268s（MAE 10.1s 带内）。
3. 机制结论保持：v187 Jacobian 在 clean-room（v185 基座）官方 +721，但在
   v186 成熟基座上 LOO 门控只放行 2/24 层、净效应近零——静态 importance 槽位
   已被 pair-smooth 输出拟合占据，该移植族在 v186 上无余量。

