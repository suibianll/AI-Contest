# v162 独立分支：Linear 侧执行日志（L 代理）

> 契约：[总计划 §2/§5/§6](../docs/superpowers/plans/2026-09-06-v162-independent-linear-attention-plan.md)。
> 只写本侧目录与本日志；GPU 经 `artifacts/proxy_v3/v162-independent/gpu.lock` 排队。
> 零点 v162 `56101559...C000A`（官方 1001/146s）。本地最高参考（eval-v3 linear
> 六 shard，n=336）：**0.6367996275406851**（v189 Linear 侧，
> `artifacts/proxy_v3/linear_static_actorder_hdiag_full/candidate/manifest.json`）。

## L0 基线建立（2026-09-06，DONE）

- baseline 副本 `workbench/v162_linear/baseline/solution.py`，SHA 核验与零点一致
  （`56101559...C000A`）。
- 运行：linear-only 六 shard，candidate=零点副本、baseline-solution=归档零点。
  输出 `artifacts/proxy_v3/v162-independent/linear/l0-baseline-zero/v162-linear-l0-baseline/`。
- 结果：336 cases，mean/median/min/max **全部精确 0.0**；api_total 2.46s；checks
  全过（finite、case 身份唯一、覆盖完整）。GPU 锁正常获取/释放（pid 46400）。
- 状态：**DONE**（零点确认）。

## L1 l1-v160-stack-recovery（2026-09-06，DONE / RECOVERY 确认）

- 机制卡：`workbench/v162_linear/mechanism.md`（RECOVERY：v160 Linear 栈整体迁入，
  官方锚 v163 4587/202s）；config：`workbench/v162_linear/config.json`。
- 候选源码：归档 v163 原位只读引用（SHA `3352BDEC...3EB612`；钩子禁止 Bash 复制
  源码文件，L2 起自研候选经 Write/构建脚本落盘）。
- 冻结侧 control（`workbench/v162_linear/control_check.py`，shard0 真实输入）：
  **0 failures**——attention 四 API 与 v162 逐位一致（五字段/state/输出）、逆序重放
  一致、28/28 weight state 非空（gram/h_inv/block_smooth 等，机制可达）。
- 六 shard ID 评测（baseline=v162 零点）：
  - **mean 0.6281816416010996 / median 0.6240599317113444**，n=336，min −0.1361，
    非正 case 2；
  - 分 role：q 0.7489 / k 0.7759 / v 0.7680 / o 0.5203 / fc_gate 0.5511 /
    fc_up 0.5025 / proj 0.5305；
  - 相对 v162 零点累计 Δ0 = +0.6282（RECOVERY 达成）；相对本地最高
    （v189 Linear 侧 0.6367996275406851）**−0.008618**，未超过；
  - 与 v189 Linear 侧差值集中在 proj（−0.034）与 o（−0.014），与 rank-2/块序
    机制在后两版的作用面一致；
  - api_total 275.5s（本地，W_calib 168 + dyn_act 336）；输出
    `artifacts/proxy_v3/v162-independent/linear/l1-v160-stack-recovery/id/`。
- L1_total/L1_negative/OOD/时间预测：本版为 RECOVERY 基线，门禁全套在 L2 起
  每版执行；本地 api_total 不能换算官方时间。
- 状态：**DONE（RECOVERY_ONLY）**；未超本地最高，无晋级动作。

## L2 l2-rank2-recovery（2026-09-06，DONE / RECOVERY 确认）

- 机制：v160 栈 + fused rank-2 残差重分布（预注册于 mechanism.md 阶梯；官方锚
  v182 17598/273s，其 Linear 侧 = v160 栈 + rank-2，v186/v189 未再改 Linear 侧
  除块序外）。构建：`build_l2_candidate.py l2`——v163 + v163→v182 diff 的五个
  Linear hunk（`71/8155/8218/8429/8473`），不应用 attention hunk（`365/8761/
  10188`）与标准块删除 hunk（`10295`）；自检 `v163+full diff == v182` 逐字节
  通过。SHA `AFD6F116C70D4E56B0746A5C6D4203D2C8F6A82C78C745AEE3173FC5F2BE361A`，
  文件 `workbench/v162_linear/candidate/l2_solution.py`。
- 隔离检查：rank-2 state 完整（`rank1_u/v`+`residual_u/v`，`[L-R2] rank2
  reachable=1 vtu_cross_max=1.9e-08`），动态五字段正常。
- 冻结侧 control：**0 failures**（与 L1 同一标准块，逐位一致确认）。
- 六 shard ID 评测（baseline=v162 零点）：
  - **mean 0.6327615862888384 / median 0.6276666933955108**，n=336，非正 case 0；
  - 单步 Δ vs L1 **+0.004579944687738813**，集中在 proj（0.5305→0.5644），
    与 rank-2 的宽形状作用面一致；
  - 相对本地最高 0.6367996275406851 仍 **−0.004038**；
  - api_total 406.1s（本地）；输出
    `artifacts/proxy_v3/v162-independent/linear/l2-rank2-recovery/id/`。
- 状态：**DONE（RECOVERY_ONLY）**；OOD 门随 L3 成对运行补齐。

## L3 l3-static-actorder-recovery（2026-09-06，DONE / RECOVERY 确认）

- 机制：+ v189 静态部署 Hessian activation-GPTQ 64-block 块序（预注册阶梯第 3
  步；官方锚 v189 17616/275s step_gain +17）。构建：`build_l2_candidate.py l3`
  ——L2 + v182→v189 diff 的纯追加尾部块（old_start 10783，+213 行，追加块与
  v189 对应块逐字节一致已验证）；不应用 `_DYNAMIC_OFFSETS` +4 hunk（old 498，
  v186 attention 侧改动，冻结纪律要求排除）。SHA
  `7A89A87B146F06C6E8CA27B91B2798474385C1A496FAC3B0C96DAB51F3EEED3A`，文件
  `workbench/v162_linear/candidate/l3_solution.py`。
- 冻结侧 control：**0 failures**。
- 六 shard ID（baseline=v162）：**mean 0.6367051523985937 / median 0.630845**，
  n=336，非正 case 0；单步 vs L2 **+0.003943566109755325**；相对本地最高
  **−9.45e-05**。api_total 391.8s。输出
  `artifacts/proxy_v3/v162-independent/linear/l3-static-actorder-recovery/id/`。
- 状态：**DONE（RECOVERY_ONLY，单机制归因步）**。

## L4 l4-v189-linear-exact（2026-09-06，DONE / RECOVERY 完成，逐位复现）

- 定位修正：`_DYNAMIC_OFFSETS` 是 v160 栈内 Linear/Attention 共享常量；本候选的
  活动 Attention 路径是追加标准块，**不读该常量**，因此 v186 的 +4 码窗在本侧
  文件中只影响 Linear 动态路径。v189 的本地最高参考（eval-v3 0.6368）在 +4 窗
  下测得；精确复现必须包含。
- L4 = L3 + 该一行常量（取自 v189 原文）。结构验证：`diff(L4, v189)` 全部
  hunk 只落在 attention 常量/代码区与标准块尾部——**L4 的 Linear 侧与 v189
  逐字节一致，Attention 侧与 v162 标准块逐字节一致**。SHA
  `ACB16F764DB80EDA94EB77FE497A7965C716B527571C241C79B2319529FF5263`，文件
  `workbench/v162_linear/candidate/l4_solution.py`。
- 冻结侧 control：**0 failures**（shard0 真实输入、五字段/state/输出、逆序重放）。
- 六 shard ID（baseline=v162）：**mean 0.6367994885532791 / median
  0.6311913371890324**，min 0.15793560695103903、max 0.9333204049812948——
  median/min/max 与 v189 Linear 侧参考**逐位相同**，mean 差 **−1.4e-07**
  （聚合浮点噪声级）＝逐位复现；非正 case 0。
  - 分 role：q 0.7535 / k 0.7789 / v 0.7714 / o 0.5337 / fc_gate 0.5537 /
    fc_up 0.5023 / proj 0.5642（与参考一致）；
  - 单步 vs L3 +9.43e-05（+4 窗的 Linear 侧贡献）；相对 v162 零点累计
    **+0.6368**；
  - api_seconds（本地）：W_calib 289.5、dyn_act 109.2；输出
    `artifacts/proxy_v3/v162-independent/linear/l4-v189-linear-exact/id/`。
- **时间门注意**：用本机本次实测代入时间模型 ≈ 170.3+33.3+80.2+标准注意力项
  ≈ 283s，名义上高于 280s 提交门；但同机制 v189 完整栈官方实测 275s，而本侧
  用标准 Attention 替换优化 Attention（A_calib 项更小），官方时间预期显著低于
  v189。本地跨会话 CUDA 计时漂移（同机制 hdiag 运行 W_calib 256s vs 本次
  289s）使单次本地外推不可靠；fresh default 实测与官方锚点共同记录，提交
  裁决归协调者。
- 状态：**DONE（RECOVERY_ONLY，阶梯终点）**。本地精度追平已知最高；要“超过
  本地最高”进入 L5+ 新机制阶段。

## L4 fresh default（2026-09-06，时间门 PASS）

- compat 后端完整六 API（168 Linear + 120 Attention，标准 Attention）：
  **linear_mean 0.6402583244298936**（与 v189 文档值 0.640258324430 一致——
  第三次逐位复现确认），attention 0.0，overall 0.3735（含标准 Attention，
  按 §3.1 不与完整父 Overall 比较）。
- api_seconds：W_calib 292.48（168 次）、dyn_act 63.54（168 次）、A_calib 0.0、
  dyn Q/K/V 0.27/0.24/0.23；api_total 356.77s、wall 383.03s（本地）。
- **时间模型**：`T ≈ 170.3 + 0.115×292.48 + 0.694×0.0 + 0.734×63.54 − 1.58×0.74
  = 249.4s < 280s` —— 提交门通过。
- 混合坐标四臂分解（E10/E01）对变换坐标栈无归因意义（AGENTS §5.4 已警示），
  仅记录不使用。
- 结果报告：`workbench/v162_linear/result-ladder.md`。

## L 链 OOD 成对门禁（2026-09-06，全链 PASS）

| 版本 | ID mean | OOD mean | gap=ID−OOD | \|Δgap\| vs 直接父 |
|---|---|---|---|---|
| L1 | 0.628182 | 0.642120 | −0.013938 | — |
| L2 | 0.632762 | 0.644909 | −0.012148 | 0.001791 ✅ |
| L3 | 0.636705 | 0.648661 | −0.011955 | 0.000192 ✅ |
| L4 | 0.636799 | 0.648735 | −0.011935 | 0.000020 ✅ |

所有版本 OOD 均值高于 ID（OOD 窗口整体略易），无退化信号；各步 |Δgap| ≤ 0.01
通过。输出 `artifacts/proxy_v3/v162-independent/linear/<version>/ood/`。

<!--
统一状态字段（总计划 §6）：
side/run_id/baseline_sha/parent_sha/candidate_sha/mechanism/config_sha/panel/
focus_gain/step_gain/strong_control_gap/L1_total/L1_negative/OOD_gap/control/
coverage/reachability/api_times/time_prediction/official_score/official_time/status
-->
