# v162_linear L4：v189 Linear 侧逐位复现（侧隔离包）

> 日期：2026-09-06。分支：v162 独立 Linear（L 代理）。
> 契约：[总计划](../docs/superpowers/plans/2026-09-06-v162-independent-linear-attention-plan.md)
> + [L 任务书](../docs/superpowers/plans/workpackages/v162-linear.md)。
> 分支前缀 `v162_linear_`，全局版本号/根替换/组合由协调者负责。

## 官方结果（2026-09-06 用户回传）

- **official score：4607 / 247s**（提交 SHA 即本包 solution.py
  `ACB16F764DB80EDA94EB77FE497A7965C716B527571C241C79B2319529FF5263`）。
- **C_L = 4607 − 1001 = 3606**（相对 v162 零点的官方累计贡献）。
- 时间 247s：通过 `<280s` 提交门与 `<300s` 硬限；时间模型预测 249.4s，
  实测误差 −2.4s（模型有效）。
- 官方锚点判读：
  - vs v163（同一 L1 机制，C_L=3586）：**+20** —— rank-2 + 块序 + 码窗
    三步已验证增量的官方合计，与各自完整父锚点（v182 +1 / v189 +17 /
    v186 +1，合计 ≈ +19）在官方 ±1~4 噪声带内一致，无异常；
  - vs 强对照 v166（rank-1，4590）：**+17**；
  - 官方正裁决成立：本包为当前已知官方最优的 Linear 侧隔离包
    （candidate Linear + standard Attention 形态）。
- 交互检验待 Attention 侧官方结果回传后由协调者执行
  （S_additive = S_L + S_A − 1001）。

## 源码与身份

- **solution.py SHA256：`ACB16F764DB80EDA94EB77FE497A7965C716B527571C241C79B2319529FF5263`**
- 零点父：v162 `56101559...C000A`（官方 1001/146s）；分支父链 L1→L2→L3→L4。
- 机制（RECOVERY，逐字节移植自已归档源）：v160 Linear 校准栈 + rank-2 残差
  重分布（v182 Linear 侧）+ v189 静态 Hdiag activation-GPTQ 64-block 块序 +
  `_DYNAMIC_OFFSETS` +4 码窗（v186 常量）。结构验证：`diff(L4, v189)` 全部
  hunk 仅落在 attention 区与标准块尾部——**Linear 侧与 v189 官方父逐字节一致，
  Attention 四 API 与 v162 标准实现逐位一致**。
- 构建器：`workbench/v162_linear/build_l2_candidate.py l4`（全量 diff 自检
  `v163+diff==v182` 逐字节通过；产物可确定性重建）。

## 本地门禁（全部通过）

| 门 | 结果 |
|---|---|
| eval-v3 六 shard ID（baseline=v162） | mean **0.6367994885532791** / median 0.6311913371890324，n=336，非正 case 0 |
| 与 v189 Linear 侧参考 | median/min/max 逐位相同，mean 差 −1.4e-07（聚合浮点级）＝逐位复现＝当前本地最高 |
| 冻结侧 control | 真实 Q/K/V 输入、真实调用顺序下与 v162 逐位一致（五字段/state/输出），逆序重放一致，0 failures |
| 机制可达 | 28/28 weight state 非空；`[L-R2] rank2 reachable=1`、`[STATIC-ACTORDER-HDIAG] reachable=1` |
| OOD（候选/直接父成对） | gap −0.011935，\|Δgap vs L3\| = 0.000020 ≤ 0.01 ✅（全链 L1–L4 亦通过） |
| fresh default（compat，六 API） | linear_mean **0.6402583244298936**（与 v189 文档值一致）；attention 0.0 |
| 时间模型 | T ≈ 170.3 + 0.115×292.48 + 0.694×0.0 + 0.734×63.54 − 1.58×0.74 = **249.4s < 280s** ✅ |
| 自包含 | 脱离仓库导入检查通过（仅 math/typing/torch；六 API 合法 shape/五字段） |

## 官方结果记录

- 首测（归档时）：unregistered / NA；2026-09-06 用户回传 **4607 / 247s**（见上）。

## 证据索引

- 执行日志：`logs/execution/v162-independent-linear.md`（L0–L5 全记录）。
- 阶梯报告：`workbench/v162_linear/result-ladder.md`（含强对照 v166 实测
  0.6325、D_strong=1.17%、L5 新机制 REJECTED 记录、机制空间清点）。
- 评测产物：`artifacts/proxy_v3/v162-independent/linear/l4-v189-linear-exact/{id,ood}`、
  `fresh-default.json/md`、`strong-control-v166/id`。
