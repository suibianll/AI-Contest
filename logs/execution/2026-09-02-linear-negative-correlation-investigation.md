# 本地 Linear 与官方负相关调查总结报告（A1+A2+A3，2026-09-02）

> 问题：对 v84 之后 10 个有官方成绩的版本重测（`reeval5-*`，5 折协议）后，本地 Linear
> mean 与官方分数的 Pearson ≈ **−0.61**：本地把 v138–v156 的 0.58 系排到 v86/v157/v158 之上，
> 官方恰好相反（v140 系 15715–16581 全部低于 v86 16744）。本报告汇总三轮归因实验。
>
> 分步详情见：
> - `logs/execution/2026-09-02-A1-linear-negcor-role-layer.md`
> - `logs/execution/2026-09-02-A2-cross-proxy-qkv-overestimate.md`
> - `logs/execution/2026-09-02-A3-transform-off-gain-source.md`

## 1. A1：负相关集中在 q/k/v 窄层（role/layer 剖析）

- v084 = v086 = v158 的 Linear 逐 family 相同（qkv 0.6211 / o 0.4283 / fc 0.4749 / proj 0.3898）
  → 官方两次增量（v84→v86 +227、v86→v158 +117）纯 Attention，本地冻结完全复现。
- family 与官方相关（n=10）：qkv **−0.629** > o −0.625 > fc −0.572 > proj −0.200。
- lo−hi 组差（v140 系 − 高分组）：qkv **+0.125**（k +0.164 / v +0.125 / q +0.085）≫ o +0.030 > proj +0.022 ≈ fc +0.008。
- 层聚集：抬升集中在 L22(+0.167)/L14/L20/L11/L2；lo 反而更差的 L3(−0.064)/L21 恰是官方反向最小处。

**结论**：负相关主要来自 **q/k/v（窄层）的本地大幅抬升**，不是 fc/proj 的变换收益。

## 2. A2：q/k/v 抬升跨代理一致高估（四列对照）

v140−v086 逐 role delta（官方 = **−906，回归**；三个独立代理全给正）：

| role | Qwen | GPT-2 | 外部 hif4 |
|---|---:|---:|---:|
| q | +0.084 | +0.076 | +0.041 |
| k | +0.163 | +0.109 | +0.090 |
| v | +0.124 | +0.032 | +0.009 |

- 4 点 Pearson：Qwen QKV vs 官方 = **−0.61**；GPT-2 / 外部 hif4 同样把 v140 排到 v86 之上。
- **附加发现**：fc / proj 只有外部 hif4 与官方同向（fc −0.045、proj −0.015）；Qwen/GPT-2 全为正。

**结论**：q/k/v 高估是**跨代理系统性现象**（非 Qwen 特有伪影）；任何以降低 q/k/v 编码误差为目标的
手段都不能作为官方方向证据；fc/proj 机制的离线排序若用代理，优先采信外部 hif4。

## 3. A3：本地 Linear 增益几乎全部来自等价变换族（transform-off）

制作 v158 的 `LOCAL ATTRIBUTION CONTROL` 变体（切断 SmoothQuant/Permutation/block-Hadamard/CAT，
保留全部量化 refine），`--linear-only` default-panel（2 折口径）：

| 版本 | linear_mean | q | k | v | o | fc_gate | fc_up | proj |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| v158 parent（=v86） | 0.448180 | 0.583 | 0.602 | 0.617 | 0.361 | 0.409 | 0.410 | 0.155 |
| v158 **transform-off** | **0.321107** | 0.399 | 0.529 | 0.464 | 0.274 | 0.300 | 0.243 | 0.039 |

- 去掉变换后分数掉 **−0.127（−28%）**，七大 role 全部回退，无一带侥幸；
- role 回退：q **−0.184** > fc_up −0.167 > v −0.152 > proj −0.116 > fc_gate −0.109 > o −0.088 > k −0.074；
- **q/k/v 最依赖变换**（QKV 平均 −0.137），与 A1 的"q/k/v 假信号最大"完全闭环；
- 纯量化 refine（offsets/importance/GPTQ/JDRQ/C45）的真实本地水平 ≈ **0.321**。

**结论**：本地 Linear 高分的主体 = **等价变换族在本地结构上的伪收益**；Quantizer refine 本身
贡献很小。官方历史已验证变换族增量（ROAB +123 不可移植、permu/stored-scale 不迁移、0.58 系
整体不认可），因此本地 Linear 分数的可迁移性极低。

## 4. 综合结论（A1+A2+A3 闭环）

1. **本地 Linear proxy 对官方的方向判断不可用**：本地高分 = 变换族伪收益（主）+ 量化 refine（小），
   二者都未在官方取得一致证据；q/k/v 是最集中的假信号区，且跨 Qwen/GPT-2/外部 hif4 一致高估。
   > **[2026-09-04 修订]** 结论方向正确但**过窄**：不只是 Linear 侧，而是**两侧本地值都不可
   > 定量预测官方分**（LOO MAE ≈ 1108 分）。且「跨 Qwen/GPT-2/外部 hif4 一致高估」里的跨模型
   > 旁证已废弃（ρ = −0.071）。见
   > [修订清单 §1 / §9.1](../../../docs/stale-information-inventory-2026-09-04.md)。
2. **官方分数在 Linear 侧不奖励本地任何提升**：v84→v86、v86→v158 的官方增量全部来自 Attention；
   Attention 是唯一与官方高度同序（Spearman 0.85）的面。
3. **量化器编码（HiF4 E6M2+lv2/lv3+mantissa）受赛题约束不可改；机制层未到头，但提升必须官方裁决**。

## 5. 策略调整

| 项 | 更新 |
|---|---|
| 本地 Linear 分数 | 不再作为晋级/方向依据；默认标注低置信（尤其 q/k/v 与深层） |
| q/k 方向机制 | 关闭为官方方向依据（QK Smooth/K center/pair transform 家族不再叠加） |
| fc/proj 离线排序 | 若用代理，优先外部 hif4（唯一与官方同向的代理） |
| 量化 refine 族 | 无变换、结构较无关，仍是官方单变量验证的合理对象（以官方回传裁决） |
| 算法主线 | 保持：Attention 为已验证同序面；Linear 只做结构+闭式（single-pass block-Schur GPTQ） |
| parent 体系 | v158（16861/223s）不变 |

## 6. 关键资产

- 评测 JSON（5 折，仅同协议可比）：`artifacts/official_eval/reeval5-*.json`
- A3 控制文件：`artifacts/official_eval/v158-transform-off-solution.py`（LOCAL ATTRIBUTION CONTROL，非提交包）
- A3 评测：`artifacts/official_eval/a3-{v158-parent,v158-transform-off,v86}-lin.json`
- 分步日志：A1 / A2 / A3 三份 `logs/execution/2026-09-02-A{1,2,3}-*.md`