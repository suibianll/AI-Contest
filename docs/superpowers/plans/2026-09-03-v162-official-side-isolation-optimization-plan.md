# v162 基线的 Linear / Attention 官方侧向隔离优化计划

> 状态：**ACTIVE**（v165 已官方 timeout；当前计划继续执行 Linear 首发候选与一次低复杂度
> Attention 重构）
>
> 创建：2026-09-03
>
> 修订：2026-09-03（§6.1 增加预注册数值稳定界 `|v^T u| ≤ 1/2`）
>
> 官方共同基线：v162，`1001 / 146s`，源码 SHA256
> `56101559D267D962084CD67A9F9AF8EB924501B17AB408EAF676081876CC000A`
>
> 当前完整方案锚点：v160，`17532 / 232s`
>
> 当前榜首锚点：`21765 / 290s`（用户回传，源码未知）

## 1. 计划目标

后续始终从 v162 的双标准 HiF4 行为建立候选，分别替换一侧 API：

```text
Linear 候选：    candidate Linear    + standard Attention
Attention 候选： standard Linear     + candidate Attention
组合候选：       best official Linear + best official Attention
```

这样每次官方回传只包含一个方向的算法变化，不依赖本地 proxy 到官方分数的换算，能够直接得到：

1. Linear 或 Attention 相对 standard HiF4 的官方绝对贡献；
2. 新机制相对当前 v160 对应侧的官方增量和提升比例；
3. 两个候选组合后的官方交互项；
4. 新方案对 `4233` 分榜首差距的实际闭合比例；
5. 每一侧及组合的真实官方时间。

本计划降低本地晋级门槛。本地评测负责证明实现合法、机制确实执行、另一侧未被污染，并记录风险；
除接口错误、非法 state、非有限输出、机制不可达等实现问题外，不以 compact/default 的轻微负向、
median、worst case 或经验时间比例取消首次官方提交。算法是否提升由官方分数决定。

## 2. 已有官方锚点

| 记号 | 版本 | Linear | Attention | 官方分数 | 官方时间 |
| --- | --- | --- | --- | ---: | ---: |
| `S00` | v162 | standard | standard | `1001` | `146s` |
| `S10` | v163 | v160 | standard | `4587` | `202s` |
| `S01` | v164 | standard | v160 | `13945` | `204s` |
| `S11` | v160 | v160 | v160 | `17532` | `232s` |

当前两侧官方贡献为：

```text
Linear_v160    = S10 - S00 = 3586
Attention_v160 = S01 - S00 = 12944
interaction    = S11 - S10 - S01 + S00 = 1
```

`interaction=1` 只证明 standard/v160 这四个端点近似可加，不保证未来算法仍然可加；未来组合必须
用一次真实官方结果重新测量交互项。官方时间也不得按上述公式相加。

## 3. 每个官方结果的统一计算方法

### 3.1 Linear 候选

设某个隔离 Linear 候选的官方结果为 `S_L / T_L`：

```text
Linear 绝对贡献：       C_L = S_L - S00
相对 v160 Linear 增量： G_L = S_L - S10
相对侧贡献提升率：      P_L = G_L / 3586
相对 v160 恢复倍率：    R_L = C_L / 3586
```

其中 `P_L` 是后续所称的“Linear 官方优化比例”。例如 `P_L=0.10` 表示该候选在 v160 Linear
已有官方贡献基础上再增加了 10%，不是总分上涨 10%。同时记录总分口径：

```text
P_L,total = (S_L - S10) / S10
```

### 3.2 Attention 候选

设隔离 Attention 候选的官方结果为 `S_A / T_A`：

```text
Attention 绝对贡献：       C_A = S_A - S00
相对 v160 Attention 增量： G_A = S_A - S01
相对侧贡献提升率：          P_A = G_A / 12944
相对 v160 恢复倍率：        R_A = C_A / 12944
P_A,total                  = (S_A - S01) / S01
```

### 3.3 组合候选

选择官方结果最好的、未超时的 Linear 与 Attention 候选后，先形成结构性预测：

```text
S_pred = S_L + S_A - S00
```

随后只用组合候选的真实官方结果 `S_LA / T_LA` 计算：

```text
实际交互项：       I = S_LA - S_L - S_A + S00
相对 v160 增量：   G_total = S_LA - 17532
榜首差距闭合率：   closure = G_total / 4233
剩余差距：         gap = 21765 - S_LA
```

所有百分比同时报告分子、分母和绝对分数，避免只报比例造成误读。若分数缺失或 timeout，不计算
精度比例；时间只记录官方实测值，不由本地秒数换算。

## 4. 宽松但必要的本地检查

每个候选在官方提交前只要求完成以下检查：

1. 单文件脱离仓库能够导入，六个 API 名称和返回结构合法；
2. `reference_hif4.py` state 校验通过，输出无 NaN/Inf；
3. 目标侧至少一个真实 state 或输出相对 v162/v163/v164 对应父侧发生变化，排除死分支；
4. 非目标侧与 v162 standard 输出逐位一致；
5. 保存 source SHA、compact/default 描述性结果、API/wall 时间和复现命令；
6. 对 Linear 记录 W-only/A-only/Both/interaction；对 Attention 记录 Q/K/V、QK/QKV、
   probability MSE/KL；这些数据解释官方结果，不作为严格晋级线；
7. 若本地整体负向或时间很高，在 result 中标记风险，但只要实现合法且机制有明确理论目的，
   不自动取消该数学机制的首次官方测量。

以下情况才在官方前停止：导入失败、非法 state、非有限输出、目标机制没有执行、control 被意外修改，
或本机运行已经无法完成一次完整调用。修复实现错误时保持原数学规则；不得借修复改变参数。

## 5. 防止过拟合的约束

门禁放宽不等于允许围绕官方榜单调参：

- 每个候选只改变一个可解释机制，配置在读取本地 holdout 和官方结果前固定；
- calibration、local holdout 和 official 三者用途分离；本地结果只作描述和查错；
- 同一机制首次官方负向后，不扫描 threshold、rank、seed、block size、fold 权重或 role/layer 路由；
- 官方 timeout 可以对同一数学目标做一次纯复杂度重构，但必须说明输出是否等价；
- 官方正向无论大小都如实保留；是否成为主线根据绝对增量和时间决定，不设 `+300`、`+1000`
  等人为准确率门槛；
- 不重复提交相同 SHA 或逐位等价候选估计评测噪声。

## 6. Linear 路线

### 6.1 官方父侧与候选构造

Linear 的现有官方父侧为 v163：v160 Linear + standard Attention，`4587 / 202s`。新候选从
v162 单文件骨架构造，只替换两个 Linear API；四个 Attention API 保持 v162 standard 逐位一致。

首个新机制固定为**秩 1 可逆残差重分布**，在 v160 Linear 编码器之前加入：

```text
R = I + u v^T
A' = A R
W' = W R^{-T}
```

连续域保持：

```text
A' W'^T = A W^T
```

`u/v` 只从 calibration folds 与父编码残差解析构造，rank 固定为 1，不做强度、rank、role 或
layer 网格。

数值稳定界（预注册公式常数）：`R` 的可逆性由 Sherman-Morrison 分母 `1+v^T u` 决定，
`v^T u -> -1` 时 `R` 近奇异，`W R^{-T}` 变换会放大量化误差。`u/v` 构造完成后执行一次固定
投影：若 `|v^T u| > 1/2`，则 `u <- u / (2|v^T u|)`（只整体缩放、不改方向），使
`1+v^T u ∈ [1/2, 3/2]`，`R` 的特征值（`1` 的 `D-1` 重根与 `1+v^T u`）全部落在 `[1/2, 3/2]`。
该界不由 holdout 或官方结果调整。

动态路径利用：

```text
A R = A + (A u) v^T
```

避免保存或乘完整 `D×D` 矩阵。

### 6.2 Linear 算法流程

1. 从 v160 Linear 得到父量化权重和最终部署残差；
2. 对两个 calibration window 做预先固定的折分，分别计算残差稳定方向；
3. 对折间方向做符号对齐和中位聚合，只产生一个 rank-1 变换；
4. 用 Sherman-Morrison 公式计算 `R^{-T}`，变换权重后运行原 v160 Weight 编码；
5. activation state 保存 `u/v`，动态执行一次 rank-1 更新后运行原 v160 Activation 编码；
6. 构造 v162 standard Attention control，完成本地描述性评测；
7. 无实现错误即提交一次官方，使用 §3.1 计算 `C_L/G_L/P_L/R_L`；
8. `S_L > 4587` 即记为官方正向并成为新的 Linear 侧父版本；`S_L <= 4587` 则该机制 rejected，
   后续只能更换数学结构，不能调 rank/阈值。

### 6.3 Linear 复杂度

设权重形状为 `R_o × D`，动态 token 数为 `T`：

```text
calibration statistics: O(ND + R_oD)
weight transform:       O(R_oD)
dynamic transform:      O(TD)
stored state:           O(D)
```

本地时间只记录；`T_L < 300s` 与否由官方评测决定。

## 7. Attention 路线

### 7.1 第一官方测量：v165

v165 已完成候选构造：standard Linear + v161 Attention，SHA
`033E85D5DAF1A820BACDB14F9E35183C485E8DD489D118899A1AE3CB491D8C1D`。它相对 v164 只增加
Q/K Cross-Gram64 动态精化。

**官方结果（2026-09-03，用户回传）：`timeout（>300s，无分数）`。** 因此不计算
`C_A/G_A/P_A/R_A`，也不从本地结果推断官方精度。v164 的同侧对照为 `13945 / 204s`；在
官方评测稳定的已知条件下，v165 隔离增加的动态精化使总时间跨过 300s，增量相对 v164
至少约 `>96s`。该数值是超时下界，不是完整官方耗时。

v165 的本地 `+0.052502`、`106+/14-` 和 GPT-2 同号只说明候选值得测量，不用于预测官方分数。
官方只否定当前动态实现的时间可行性，没有给出精度裁决。按预注册只允许一次保持 Gram 数学
目标不变的低复杂度重构；不重试 v165，不缩原路径的 sweeps，也不围绕阈值或块数调参。

### 7.2 timeout 后的低复杂度版本

v165 已 timeout，下一 Attention 候选使用**校准编译的低秩 Gram 残差码本**，仍从 v162 构造
standard Linear control，Attention 主体以 v160 为父侧：

```text
H_Q = E[K^T K]
H_K = E[Q^T Q]
H ≈ D + U Lambda U^T, rank(U)=2
```

校准阶段把每个 64-block 的 Gram 压缩为固定 rank-2 state；动态阶段只在固定五个合法码本候选
中一次向量化选择，不运行三轮坐标 sweep、完整 Gram contraction 或逐坐标 Python 循环。

算法流程：

1. 冻结 v160 的 Q/K/V transform、center、importance 与 hierarchy；
2. 在最终部署坐标按 calibration folds 计算 `H_Q/H_K`；
3. 固定 rank 2，折间做符号对齐和中位子空间聚合；
4. 生成 `parent、±u1、±u2` 五个合法相邻码字模板；
5. 动态一次批量计算 `D + U Lambda U^T` 近似损失并选择模板；
6. standard Linear 和 V control 保持不变；
7. 完成本地合法性/可达性检查后提交一次官方；
8. 相对 v164 使用 §3.2 计算官方 Attention 优化比例。

### 7.3 Attention 复杂度

固定 block `B=64`、rank `r=2`、候选数 `K=5`：

```text
calibration Gram: O(FNDB)
low-rank compile: O(FDB^2)
dynamic encode:   O(KrTD) = O(TD)
stored state:     O(rD + D)
```

该复杂度说明只用于设计审计；是否满足 `<300s` 仍由官方结果决定。

## 8. 组合与最终决策

Linear 和 Attention 始终单独获得官方结果后再组合：

1. 每侧选择官方分数最高且未 timeout 的候选；
2. 构造一个单文件组合版本，先做六 API 合法性和非目标污染检查；
3. 本地完整 panel 只记录集成交互，不设准确率晋级线；
4. 提交一次官方，实测 `S_LA/T_LA`；
5. 使用 §3.3 计算实际交互、相对 v160 增量、榜首闭合率和剩余差距；
6. `S_LA > 17532` 即成为新的完整官方父版本；否则继续保留各侧官方结果，不伪造可加收益。

即使一侧官方没有提升，另一侧仍独立继续，不因本地或另一侧结果提前关闭整个双路线计划。

## 9. 固定执行顺序与产物

1. 锁定 v162/v163/v164/v160 的源码 SHA 与官方结果，不重复提交；
2. **已完成**：v165 官方 `timeout（>300s，无分数）`；不计算 Attention 精度比例；
3. 实现并隔离验证 Linear rank-1 候选，保存源码、JSON、report、日志和 SHA；
4. 提交 Linear 候选，登记 `C_L/G_L/P_L/R_L` 与时间；
5. **待执行**：实现一次低秩 Gram 码本重构；若仍 timeout，当前 Gram 动态目标关闭；
6. 两侧出现新的官方最好版本后构造一次组合候选并提交；
7. 每次官方回传后更新本计划、`docs/current-solution-status.md`、`solutions/README.md` 和独立日志；
8. 每次实质更新后运行 `git diff --check`、提交、push，并核验工作区。

每个候选 result 必须包含：父侧版本、standard control 来源、唯一机制、源码 SHA、本地 scope、
官方分数/时间、四个侧向指标、状态和下一决策。未回传官方结果时写 `unregistered / NA`。
