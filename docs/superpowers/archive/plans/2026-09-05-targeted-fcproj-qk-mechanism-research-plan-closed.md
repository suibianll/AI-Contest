# fc/proj 与 Q/K 定向机制研究计划（离线解剖 → 条件注册）

> 状态：**COMPLETED-CLOSED（2026-09-05 归档）**。S1 结构解剖完成，结论：目标对象
> （fc/proj、Q/K）剩余误差为 3-bit mantissa 网格主导（round-limited）+ 约 4.6% 组内
> 离群饱和（次要），**未发现新的合法编译目标**；按 §2 结束分支，S2/S3 未触发，无版本号、
> 无父变更、无官方提交。S1 报告：`logs/execution/2026-09-05-targeted-fcproj-qk-autopsy.md`。
> ACTIVE 期目标：在 **P3 官方证明收益所在的对象**（Attention Q/K、Linear fc/proj 大形状）与
> **P1 证明误差集中处**（深层）上，离线解剖剩余量化误差的**结构成因**，判定是否存在
> **一个可编译进合法 HiF4 五字段的固定规则**；有则注册单候选并走严格本地门禁 + 一次官方
> 验证，无则记录关闭。不承诺超过 v186，不把本地误差下降换算成官方分。
> 父版本：v186（17599/272s，SHA F8495DCA…7EB8）；v180（17597/242s）为时间预算父。
> 已用尽并关闭：坐标等价变换族（连续域零偏差）、scale/lv2/lv3 求解器（P2 R2/R3 饱和）、
> mantissa 网格（合法不可调）、rank-3/系数/fold、scale 窗口 +4 邻域、Jacobian importance
> 移植、v183 coverage、full64/Householder、V multiplier、动态 per-call Gram、
> A4/L4/C1、v185 balance/gamma/refine。本计划**不重启上述任何家族**，也不扫描邻域。

## 1. 依据（已归档证据，不重复运行）

| 证据 | 结论 | 对本计划的约束 |
|---|---|---|
| P1 同坐标分解 | Attn Q/K 单侧量化误差≈V 的 2.5–2.8×；Linear W 侧≈X 侧 1.9×（proj 4.4×）；深层误差大 | 解剖对象只取 Q/K 与 fc/proj、并按层切分 |
| P2 放宽诊断 | R1（mantissa 连续化）唯一有系统余量（Q/K −82%、W −77%）且 100% 同号；R2/R3（scale、lv2/lv3 连续化）无余量 | 剩余误差 = 3-bit mantissa 网格 + 块级 scale 结构共同作用；需区分二者在目标对象上的占比 |
| P3 官方 | Attention 增益 85% 在 Q/K（C_QK=11009）；v160 Linear 增益 100% 在 fc(W2)+proj(W3) | 若出现可编译规则，只允许作用在 Q/K 或 fc/proj；q/o 方形与 k/v 窄桶零收益，不注册 |

## 2. 阶段与验收

| 阶段 | 核心问题 | 产物 | 进入下一阶段条件 | 状态 |
|---|---|---|---|---|
| S0 冻结 | 父、artifact、工具是否可复现 | 复用 manifest + v186 校准缓存 | 全部 identity 命中 | TODO |
| S1 结构解剖 | 目标对象剩余误差是 clip 受限还是 round 受限？块内幅度分布是否健康？ | 逐 (layer, operand/role) 表：clip/high/low/zero 分数、r 分位数、round 上界 | 表完成且无未捕获异常 | TODO |
| S2 候选注册 | S1 是否给出唯一合法编译目标 | 机制卡、单文件候选、父子/OOD 报告 | 合法 + `Δmean>0 且 L1<0.02` + ID/OOD `|Δ|≤0.01` + `T_pred<280s` | CONDITIONAL |
| S3 官方验证 | 是否真的超过 v186 单侧/组合 | 官方 result、SHA、保留/拒绝 | 官方回传后判定 | CONDITIONAL |
| 结束 | S1 无可编译规则或候选官方非正 | 关闭记录 | — | — |

S1 为**纯离线只读研究**，不产生版本号、不提交、不写默认时间模型。S2/S3 仅在 S1 出现
可编译规则后触发；否则本计划按结束分支记录并归档。

## 3. S1 解剖方法（执行定义）

对象：fc_gate/fc_up/proj 全部 24 层权重（W2/W3 桶，P3 官方正收益）与 Attention Q/K
全部层（P3 官方 85% 收益），参考 P1 的六 shard 面板（2 windows/layer）。

对每个权重 state 或每个 Q/K dynamic case：

1. 用已验证的同坐标镜像重建**编码前连续张量**（权重：`coordinate_diagnostics.linear_weight_continuous`；
   Q/K：`solution._attention_state_transform_dense`），与已量化 params 配对。
2. 展开逐元素 `r = |x| / (scale_factor·lv2·lv3)`（复用放宽解码的 denom 结构），统计：
   `clip_frac`（r≥1.75，E6M2 向下取整导致的饱和）、`high_frac`（r≥0.75）、
   `low_frac`（0<r<0.5）、`zero_frac`、r 的 p50/p90/p99/max、`mean_r²`。
3. round 受限上界估计：若每块以 mantissa 网格 0.25 均匀量化，输出误差下限
   `E_round² ≈ mean(denom²)·0.25²/12`（同坐标下按 P1 方式映射到输出空间再解读）。
   与 P1/P2 已存的 arm 误差对照。
4. 输出每对象表 + 汇总；判定：clip 显著→可考虑块级 scale/层级结构（但 P2 R2/R3 已饱和，
   需解释为何合法求解器遗漏）；纯 round 受限→无合法杠杆，结束分支。

产物：`artifacts/proxy_v3/targeted-autopsy-20260905/<run>/`（manifest/表/md，不入库）；
日志 `logs/execution/2026-09-05-targeted-fcproj-qk-autopsy.md`。

## 4. 纪律与失败处理

- 任何 S1 数字都不用于注册候选的"强度"论证，只用于是否**存在唯一合法编译目标**。
- 不得借 S1 重开邻域扫描；若候选需要扫描任何超参/邻域，视为不支持注册。
- 时间：S1 无 API 时间统计（全离线）；S2 若触发，仅目标侧 shard 配对 + 一次新鲜 default
  计时（`T_pred<280s` 才提交）。
- 官方提交：每次只 1 个候选、预注册配置固定，官方失败即关闭，不调参重试。
