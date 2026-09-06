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

<!--
统一状态字段（总计划 §6）：
side/run_id/baseline_sha/parent_sha/candidate_sha/mechanism/config_sha/panel/
focus_gain/step_gain/strong_control_gap/L1_total/L1_negative/OOD_gap/control/
coverage/reachability/api_times/time_prediction/official_score/official_time/status
-->
