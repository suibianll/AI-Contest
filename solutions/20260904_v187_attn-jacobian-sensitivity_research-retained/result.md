# v187 Attention Jacobian 敏感度加权 HiF4

> 状态：RESEARCH RETAINED / NOT SUBMITTED
>
> Parent：v185 clean-room（官方 8446 / 165s）。Linear 与 V 不变；唯一改动是把解析
> Attention 输出 Jacobian 对角敏感度用于 Q/K HiF4 坐标加权，并经 leave-one-fold-out gate。

## 机制

对 causal/non-causal Attention 输出计算 Q/K 一阶 Jacobian 对角敏感度，压缩为 KV-group
共享的 `KV-head×64` importance；每 fold 归一化、跨 fold median、log-space 1/4 收缩到
identity，最后由 leave-one-fold-out 最终输出 MSE gate 决定是否部署。在线只有固定 importance，
Linear 与 V 完全不变。

## 本地结果

- 六 API smoke、HiF4/state 合法性：通过；
- compact：mean `0.694529177`；相对 v185 `+0.007988`，`1+/0-/3=`；
- default：mean `0.418953857`；相对 v185 `+0.015186535`、L1 `0.016199247`，
  `32+/3-/85=`；
- 7/24 层启用：3/5/7/14/15/17/20；importance 范围 `0.5491–2.0`；
- 相对 v186：`-0.333219550`、`4+/116-/0=`，median MSE ratio `2.261583`；
- Attention API `26.770s`，比 v185 增加 `3.171s`。
- 评测器与参考 codec 定向测试：`43 passed`；全仓测试另有 3 个历史测试依赖根文件已删除的
  `_choose_boat/_encode_rows` 私有函数，不属于 v187。

源码 SHA256：`086535FB4205703524C5DF2378CF2557B7F4652DF03E6FA201C074F2094F8F65`。

## 裁决

该解析机制通过 R2，说明最终输出敏感度比等权重建更合理；但它无法弥补 v185 与成熟 v186
之间的机制差距。保留为 clean-room RESEARCH RETAINED 父，不提交、不占配额、不扫描参数邻域。

## 官方

- Score：NA
- Time：NA
- Status：not submitted
