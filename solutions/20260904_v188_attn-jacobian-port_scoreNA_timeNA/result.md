# v188 Attention Jacobian 敏感度移植（v186 + v187 机制）

> 状态：OFFICIAL PENDING — scoreNA / timeNA（提交队列，第 9 个配额 9/10）
>
> Parent：v186（官方 17599 / 272s，SHA `F8495DCA...7EB8`）。唯一改动：把 v187 的
> attention 输出 Jacobian 一阶敏感度 importance 作为校准最终步移植进 v186，
> LOO 门控通过后替换 Q/K state importance；在线路径零新增计算。

## 机制

在 logit gain 之后、返回 state 之前：对最终变换坐标系（multiplier/permutation/
center/rotation/block-smooth/pair_transform 全链）上的校准 Q/K 计算 attention
输出一阶 Jacobian 敏感度（Q 侧含跨 key softmax 耦合项），压缩为 KV-group 共享的
KV-head×64 importance：逐 fold causal/non-causal 0.5 平均、跨 fold median、
log-space 1/4 收缩、[0.5, 2.0] clamp；leave-one-fold-out 部署 MSE 门
（median 相对增益 > 0.0005 且 worst > −0.01，v187 预注册常量，无邻域扫描）。
V 与 Linear 完全不变。

## 本地结果

- 六 API smoke、py_compile、脱离仓库单文件导入：通过；
- compact 4 哨兵层：no_effect（0/0/4，门控全拒），校准 API +1.18s；
- default 120（vs v186）：Δmean **+0.000426**、L1 **0.001114**，6+/4−/110=；
- 门控 2/24 层通过（L12、L22——恰为无 pair_transform 的层；约 18 层的 importance
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

- Score：NA（待回传）
- Time：NA（待回传）
