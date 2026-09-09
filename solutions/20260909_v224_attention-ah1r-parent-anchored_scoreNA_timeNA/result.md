# v224 A-H1R 部署父状态锚定阈值事件（2026-09-09）

- run_id: `attention-ah1-parent-anchored`
- parent: 根 `solution.py` SHA256 `56dc805d6e5a3aef896db8021045740292735725d688b48e3d4393e55efcb2bd`（未修改）
- candidate: `solution.py` SHA256 `8329676485cc9ecb8d4ed2259812dd6d7015da97f3ec241a2a68d715183ec019`
- official_status: `TIMEOUT(>300s)`（用户回传，2026-09-09）；只关闭该实现
- 机制：v223 A-H1 从最后一步 Adam 更新前的 `theta_pre/center_pre` 生成事件，`t=0` 不是部署父状态。
  v224 改为：(1) 读取 gate 选中的**实际部署** `R_parent/center_parent`（identity 分支为 identity/zero）；
  (2) 在部署的**旋转前坐标系**重算 calibration fold 的 mean output gradient；
  (3) 用切空间方向 `S = skew(R_parent^T G_R)`、路径 `R(t) = R_parent @ cayley(t S)`，center 固定。
  由于 `d cayley/dt|_0 = -2 S`，路径在 `t=0` 的导数为 `-2 R_parent S`，是 `G_R` 的下降方向。
- 同时移除/旁路根中四处宽泛异常（两处 encoder 应用回退、两处 calibration 回退），并在校准中
  断言 `a2_arm != fallback`、且 state 含 learned rotation/center 时动态路径确实应用对应变换。

## 验证（`verify.py`，CUDA）

- Cayley 导数 `d cayley/dt|_0 = -2S`：rel `3.08e-05`；Cayley 正交误差 `2.38e-07`。
- 边界时间已知样例：数量/符号/排序全部匹配，无负时间泄漏。
- 合成校准：六 API 导入、`reference_hif4.validate_state` 通过、动态 Q/K/V 输出有限。
- `t0_matches_deployed_parent = true`；`pretransform_matches_deployed_encode = true`。
- 全部检查通过，见 `verification.json`。

## shard0 前证明（真实 4B layer 0）

- `ah1r_t0_identical = 1`：`R(0)=R_parent` 的 Q/K 五字段与最终输出与部署父逐位一致。
- 事件：raw `30,004,207`、dedup `24,223,402`，取最早 8 个不同事件并 `nextafter`。
- 相邻槽 changed-code 不同：Q `[1,70,70,138,76,76,134,74]`、K `[1,2,1,5,3,4,3,3]`。
- **与 v223 的关键差异**：changed-code 是小范围逐项可解释的翻码，不再是最后一步 Adam 回退造成的
  整步回退（v223 为数百万级）。

## 六 shard 结果（eval-v3 / 4B / attention-only / CUDA）

| shard | layer 父 arm | 接受事件 | fold parent | t0 | paired mean | +/-/0 |
|---:|---|---:|---:|---:|---:|---|
| 0 | rotation | 8 | 0.482951 | 1 | `-0.000004` | 4/5/3 |
| 1 | rotation | 6 | 0.487487 | 1 | `+0.000002` | 2/6/4 |
| 2 | identity | 6 | 0.409234 | 1 | `+0.000001` | 5/3/4 |
| 3 | rotation | 8 | 0.608009 | 1 | `+0.000008` | 5/3/4 |
| 4 | rotation | 8 | 0.884483 | 1 | `-0.000001` | 2/5/5 |
| 5 | rotation | 4 | 0.322082 | 1 | `+0.000001` | 3/3/6 |

- 六 shard 等权聚合 `≈ +1.2e-6`：6/6 层产生合法接受状态、部署输出真实变化，但 hard-output 效应
  在本地噪声底附近（单 shard |delta| ≤ 8e-6）。shard0 holdout 记录为转劣，仅记录不否决。
- 本地数值不换算官方分数；官方 `300s` 与官方总分为唯一裁决。

## 裁决

- 合法、可达、非等价（6/6 层翻码），按活动计划 §3 归档为 v224 并交官方独立裁决。
- **官方回传 `TIMEOUT(>300s)`**：只关闭该计算实现；与 v223 同成本类（约 3000 万次事件生成）。
  同成本类事件搜索重试前必须先降校准成本，不缩事件数/步长/窗口/seed 重试。

## 证据位置

- workbench：`workbench/full_solution/attention-ah1-parent-anchored/`（`ah1r_code.py`、`build.py`、`verify.py`、`config.json`、`verification.json`）
- 六 shard：`artifacts/proxy_v3/ah1r-sixshard-v2/`
- 官方回传日志：`logs/execution/2026-09-09-v224-official-timeout.md`
