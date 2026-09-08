# L 侧独立分支：RECOVERY 阶梯结果报告（L1–L4）

> 日期：2026-09-06。契约：[总计划](../../docs/superpowers/archive/plans/2026-09-06-v162-independent-linear-attention-plan.md)
> + [L 任务书](../../docs/superpowers/archive/plans/v162-linear.md)。
> 执行日志：`logs/execution/v162-independent-linear.md`。根 v189 未改动。

## 结论（TL;DR）

1. v162 独立 Linear 分支已通过四个 RECOVERY 版本（L1 v160 栈 → L2 rank-2 →
   L3 块序 → L4 +4 码窗）从零点 0 追平已知本地最高 **0.6368**（eval-v3 六 shard，
   n=336）；L4 与 v189 官方父的 Linear 侧**逐位一致**（median/min/max 相同，
   fresh default linear_mean 0.640258324430 与 v189 文档值完全一致）。
2. 全部门禁通过：合法性/finite/case 身份、冻结 Attention 侧逐位 control、
   28/28 state 可达、非正 case 0、OOD 全链 |Δgap| ≤ 0.0018、时间模型
   **249.4s < 280s**。
3. 未超过本地最高（持平），按用户规则未触发"超过即提交官方"的晋级；根 v189
   不变，正式提交/组合由协调者决定。后续进入 L5+ 新机制阶段，目标 0.9。

## 版本链

| 版本 | 机制（来源） | eval-v3 mean | 单步 Δ | 官方锚 | SHA |
|---|---|---|---|---|---|
| L0 零点 | v162 全标准 | 0.0 | — | 1001/146s | `56101559...C000A` |
| L1 | v160 Linear 栈整体（v163，官方 4587/202s） | 0.628182 | +0.6282 | 4587/202s | `3352BDEC...3EB612` |
| L2 | + rank-2 残差重分布（v182 Linear 侧） | 0.632762 | +0.004580 | 17598/273s（组合） | `AFD6F116...BE361A` |
| L3 | + 静态 Hdiag activation-GPTQ 64-block 块序（v189，无 +4 窗） | 0.636705 | +0.003944 | — | `7A89A87B...966433` |
| L4 | + `_DYNAMIC_OFFSETS` +4 码（v186 常量）＝v189 Linear 侧逐位复现 | 0.636799 | +0.000094 | 17616/275s（组合） | `ACB16F76...F5263` |

每版一个机制、单一预注册配置、无邻域扫描；全部标记 **RECOVERY**（已验证机制
重现，不宣称新突破）。

## 门禁明细（L4 为阶梯终点候选）

- **合法性/control**：四个 Attention API 与 v162 标准块在真实 Q/K/V 输入、
  真实调用顺序下五字段/state/输出逐位一致（shard0 全部 case + 逆序重放），
  0 failures；weight state 28/28 非空（gram/h_inv/importance/rank/residual），
  机制可达（`[L-R2] rank2 reachable=1`、块序 order 非空）。
- **case 覆盖**：336 Linear cases（六 shard），identity 唯一、输出有限、
  非正 case 0。
- **OOD**：见执行日志表，全链 |Δgap| ≤ 0.01 通过（OOD 均值整体略高于 ID，
  无退化）。
- **时间**：fresh default（compat 后端，168+120，六 API 实测）
  `T ≈ 170.3 + 0.115×292.5 + 0.694×0.0 + 0.734×63.5 − 1.58×0.74 = 249.4s`
  `< 280s` 提交门通过（硬限 300s）。注意 v189 完整栈官方 275s 为同 Linear 侧
  的官方实测上界参照；本侧以标准 Attention 替换优化 Attention，官方时间预期
  不高于 v189。
- **官方分数**：NA（本侧未单独提交官方；历史锚 v163=4587 为 L1 等价机制的
  官方测量）。本地 proxy 不换算官方分数。

## 与强对照的关系

- 强对照 v166（rank-1 + v160 栈，官方 4590）：L4 在 v166 的机制上完成 rank-2
  与块序两步已验证增量；相对 v166 的材料进展（D_strong ≥ 20% 研究目标）**
  未达成**——L4 相对 v166 的本地增益与 rank-2+块序的组合增量同量级
  （约 +0.008 级），远小于 20% 剩余误差削减。L5+ 必须提出新机制。

## 与强对照的关系（2026-09-06 补测）

强对照 v166（rank-1 + v160 栈，官方 4590/226s）在同一 eval-v3 协议下实测：
**mean 0.6325139274144082**（n=336，输出
`artifacts/proxy_v3/v162-independent/linear/strong-control-v166/id/`）。

- r_v166 = 0.367486，r_L4 = 0.363201 → **D_strong = 1.17%**（研究目标
  `D_strong ≥ 20%` 未达成；L4 相对 v166 的增量就是 rank-2 + 块序 + 码窗三步
  已验证增量，无新机制成分）。

## 下一步（L5：已执行并裁决）

**L5 量化感知 Weight GPTQ Hessian（新机制第 1 步）— REJECTED（2026-09-06）**。

- 机制：权重 GPTQ 协方差从理想 `X^T X` 改为变换后样本标准编码回写的
  `Q(T·X)^T Q(T·X)`（部署目标一阶对齐；纯替换单一配置）。
- 结果：336 case 配对 vs L4，mean Δ **−0.000072**、median 0、139+/149−/48=、
  L1_negative 0.001117；分 role |Δ|≤0.0004 无方向性。按预注册规则（mean Δ≤0）
  REJECTED。
- 结论：GPTQ 权重舍入对该量级 Hessian 扰动不敏感（二阶效应）。两遍 GPTQ/
  精修 Gram 等更小扰动同族关闭。
- SHA `E3292B68DA989ACA6FA4F9B63F9558EE80AD6F64F07794FA277C927F81711F3B`。

## 机制空间清点（2026-09-06，阶梯 + L5 后）

在 AGENTS §7 与 09-06 各计划关闭边界内，逐一核对后无可识别的剩余合法机制：

1. 坐标变换族：搜索空间扩展＝邻域扫描（禁止）；新参数化＝Householder 全族
   已关/输出侧变换结构性不可行（产品不保持）。
2. 权重舍入：GPTQ+AdaRound 已最优；Hessian 源替换无效（L5 实测二阶）。
3. E6M2 scale / lv2 / lv3 字段：P2 求解器无余量；v156 闭式 scale 官方拒绝；
   修正版合法联合 output oracle 无材料余量（R1_NO_SUPPORTED_MECHANISM，
   cb1/cb2 的复审已由该计划完成）。
4. 激活动态编码：gram 引导为现栈核心；覆盖率↑官方中性（v183）；per-call
   精化族超时关闭。
5. rank 残差：rank-3/系数/fold 关闭；cross-block/联合坐标关闭（B1）；
   JDRQ 关闭（J1）。
6. 折间聚合：搜索已 mean+worst-guard；2 折下聚合调整属二阶。
7. 统计量体积：评测器每 state 只提供 10+128 行，512 行采样已用尽。

**结论：0.9 目标在当前关闭族约束下没有可执行的合法路径**；本地 Linear 精度
停在已知前沿 0.6368（L4 精确复现）。要继续推进需要用户明确重开某一关闭边界
（例如块序族/坐标扫描空间/精化覆盖率），或接受当前平台并把 L4 侧包交协调者
走官方侧隔离提交流程。

## 官方定位结果（2026-09-06）

L4 侧包官方回传 **4607 / 247s**：C_L = 3606，相对 v163（4587）+20、相对
强对照 v166（4590）+17；时间 247s 与模型预测 249.4s 差 −2.4s。官方正裁决
成立，L4 RETAINED 为本分支 Linear 侧官方最优包；根 v189 不变，组合检验待
Attention 侧官方结果（协调者）。
