# v225 A-H3 GQA-group 局部 hard-event 坐标更新（2026-09-09）

- run_id: `attention-gqa-local-hard-event`
- parent: 根 `solution.py` SHA256 `56dc805d6e5a3aef896db8021045740292735725d688b48e3d4393e55efcb2bd`（未修改）
- candidate: `solution.py` SHA256 `eac65bd199369428bc979be8486d2cdcfc60d804c5ce06edcf3bafcb45beb767`
- official_status: `TIMEOUT(>300s)`（用户回传，2026-09-09）；只关闭该实现
- 机制：R1 用单一全局切空间方向，少数有用翻码可能被其他 KV group 的大量无关翻码抵消。A-H3 改为：
  1. 从部署根 Q/K 状态出发，冻结 V；
  2. KV group 按索引固定顺序处理一轮，每个 group 用部署父梯度的**本 group 切空间块**
     `S_g = skew(R_g^T G_g)`，分别沿 `-S_g` 和 `+S_g` 求第一个真实 Q/K hard-code 阈值；
  3. 每个 group 只比较三个状态（当前、正事件、负事件），全部走完整部署 Attention 输出；必须翻码
     且严格降低 calibration folds 聚合 loss 才接受；
  4. 接受状态作为下一个 group 的父状态；单轮、固定顺序、只取第一事件。
- 梯度只在部署父状态计算一次（各 group 的 rotation 在被处理前不受其他 group 影响），以把校准时间
  控制在根仅约 19s 官方余量的范围内。
- 同样移除/旁路根中四处宽泛异常并加 `a2_arm != fallback` 与动态应用断言。

## 验证（`verify.py`，CUDA）

- Cayley 导数 `d cayley/dt|_0 = -2S`：rel `2.73e-05`；Cayley 正交误差 `2.38e-07`。
- 边界时间已知样例通过；六 API 导入、`reference_hif4.validate_state`、动态 Q/K/V 有限输出。
- `t0_matches_deployed_parent`、`pretransform_matches_deployed_encode`、`per_group_record_length` 全部通过。
- 详见 `verification.json`。

## 六 shard 结果（eval-v3 / 4B / attention-only / CUDA，`--stop-after-nonpositive 6`）

| shard | 父 arm | 接受 group | fold parent → final | t0 | paired delta_mean | +/-/0 |
|---:|---|---:|---|---:|---:|---|
| 0 | rotation | 3/4 | 0.482951 → 0.482919 | 1 | `-1.93e-07` | 6/3/3 |
| 1 | rotation | 3/4 | 0.487487 → 0.487486 | 1 | `+1.15e-07` | 2/5/5 |
| 2 | identity | 2/4 | 0.409234 → 0.409234 | 1 | `-7.14e-06` | 2/1/9 |
| 3 | rotation | 2/4 | 0.608009 → 0.608008 | 1 | `-1.52e-06` | 3/3/6 |
| 4 | rotation | 4/4 | 0.884483 → 0.884481 | 1 | `+2.86e-07` | 3/6/3 |
| 5 | rotation | 4/4 | 0.322082 → 0.322066 | 1 | **`+1.96e-04`** | 6/3/3 |

- 基线（根，72 case）attention mean `0.533998`；候选 `0.534029`；**等权聚合 `+3.11e-05`**。
- **增益集中在 shard5（layer 22）**：该层 4/4 group 全部接受（全为负方向），changed-code 合计
  Q `8` / K `2`，calibration fold loss `0.884483 → 0.884481`。其余五 shard 在 ±7e-6 内（近零/微负）。
- 默认 `--stop-after-nonpositive 2` 会在 shard2/3 连续非正后于 shard3 提前停止，从而漏掉 shard5；
  本归档用 `--stop-after-nonpositive 6` 跑满六 shard。
- 本地数值不换算官方分数；官方 `300s` 与官方总分为唯一裁决。

## 裁决

- 合法、可达、非等价（6/6 层有 group 接受，共 18/24 group）；按活动计划 §4 归档为 v225 并交官方裁决。
- **官方回传 `TIMEOUT(>300s)`**：只关闭该计算实现；与 v223/v224 同成本类（逐 group 事件生成 +
  完整路径评估）。同成本类事件搜索重试前必须先降校准成本，不改 group 顺序/轮数/事件数重试。

## 证据位置

- workbench：`workbench/full_solution/attention-gqa-local-hard-event/`（`ah3_code.py`、`build.py`、`verify.py`、`config.json`、`verification.json`）
- 六 shard：`artifacts/proxy_v3/ah3-sixshard-v2/`；默认早停版：`artifacts/proxy_v3/ah3-sixshard/`
