# A-H1 量化阈值事件搜索 — 本地结果（2026-09-09）

- run_id: `attention-ah1-threshold-events`
- parent: 根 `solution.py` SHA256 `56dc805d6e5a3aef896db8021045740292735725d688b48e3d4393e55efcb2bd`（未修改）
- candidate: `candidate/solution.py` SHA256 `d712da36e81889c5cfaf5b5a9cc9c452b173f5e653fc6d6c6c881ae0342990da`
- official_status: `unregistered/NA`
- 评测口径：eval-v3，attention-only，cache `qwen3.5-4b-proxy-v2.pt`，algorithm-device cuda，
  `--calibration-cache-mode auto`。本地数值只判断 hard-output 优化是否真实发生，不换算官方分数。

## 实现要点

- 方向来源（兜底条款，FIX-A2 未并入根）：根原 `_a2_train_rotation` 算术不变，仅在最后一步
  Adam 更新前快照 `theta0/center0`，并在该点重算按窗口数取均值的数据梯度 + 与 trainer 相同的
  reg 项（无 clip、无 Adam），得 `g_theta/g_center`；`d=-g/max(||g||,1e-12)`，
  路径 `theta(t)=theta0+t*d_theta`、`center(t)=center0+t*d_center`，只搜 t>0。
- 事件生成：冻结父状态 permutation/offset 与父状态部署编码的 scale/lv2/lv3；解析 Cayley 导数
  `dC/dt=-(I+C)dS(I+S)^{-1}`（S(t)=S0+t·dS，rotation=base@C 分组结构），逐元素得 x0 与 dx/dt；
  对每个相邻 HiF4 magnitude 中点 `b=(j+0.5)*0.25*denom`（j=0..6）计算
  `t=(b-|x0|)/(sign(x0)*dx/dt)`（x0=0 时用 |dx|），只保留有限且严格正；float32 全局去重升序，
  评估用 `torch.nextafter(t,+inf)`。
- 固定 8 槽填最早 8 个不同事件；每层独立搜索（rotation 逐层训练的结构）。
- 真实评估：每个 t 重新走完整 `hif4_dynamic_quantize_q/k` + 当前 V 路径 + 最终 Attention
  output MSE（与官方 scorer 同形的非因果 attention）。动态 Q/K API 不含事件循环（v165 边界）。
- 选择：只按 calibration folds（calib 前 4 窗，trainer 的 prepared 窗口来源）case 等权
  normalized output MSE 聚合；t=0 父状态始终在候选集；严格更优才接受；holdout（gate 窗）只记录。
- 逐层审计字段写入 q_state/k_state（`ah1_*`），通过 `evaluator/reference_hif4.py` 合法检查。

## 验证

- `verify.py` 全过（`verification.json`）：六 API 导入；Cayley 解析导数 vs 中心差分
  rel=2.2e-4；已知小例子边界 t 符号/排序精确匹配；合成校准合法 state、有限输出；
  解析 x0 路径与部署动态编码五字段逐位一致（pretransform_matches_deployed_encode）。
- 脱离仓库单文件导入检查：通过（复制到孤立目录导入六 API 成功）。
- 父行为不变：trainer 算术未动（仅快照 + 循环后额外一趟梯度），未接受时 state 与根逐位一致
  （shard0 12 case 全部 Δ=0 证实）。

## shard0（layer 0，cache miss）

- A-H1：status=no-improvement，raw=30,123,301，dedup=24,149,545，attempted=8，accepted=0，
  fold parent=0.973900，8 个事件 fold loss 全劣（≈0.9831）。
- paired Δ：12 case 全 0（state 未变，逐位等价父）。
- calibration API 时间：`hif4_calibration_attention` 10.18s（1 层，含 A2 trainer + A-H1）；
  事件评估次数 = 8 槽 × 4 folds + parent/standard/holdout ≈ 43 次完整部署路径窗口评估。

## 六 shard（72 case，shard→layer：0→L0, 1→L1, 2→L8, 3→L15, 4→L22, 5→L5）

| layer | a2_arm | ah1 status | 接受事件 | fold parent→best | holdout parent→final | shard Δmean (+/-/0) |
|---|---|---|---|---|---|---|
| L0 | rotation | no-improvement | — | 0.9739 → 全劣 | 0.9546 → 0.9546 | 0.000000 (0/0/12) |
| L1 | rotation | accepted | #4, t=4.708e-6 | 1.0196 → 0.9719 | 0.8824 → 0.9185（劣） | +0.019407 (9/3/0) |
| L8 | identity | accepted | #3, t=1.409e-6 | 1.0000 → 0.9571 | 1.0000 → 1.0137（劣） | −0.001897 (3/9/0) |
| L15 | rotation | accepted | #3, t=1.179e-6 | 0.9992 → 0.9813 | 0.9757 → 1.0235（劣） | +0.006492 (7/5/0) |
| L22 | rotation | accepted | #4, t=4.366e-6 | 0.9944 → 0.9934 | 0.9957 → 0.9960（微劣） | +0.000555 (5/7/0) |
| L5 | rotation | accepted | #1, t=4.831e-7 | 0.9940 → 0.9699 | 0.9294 → 0.9197（优） | −0.005300 (4/8/0) |

- 总体 paired Δgain：mean **+0.003209**，median 0.0，+28 / −32 / 0-12，min −0.0422，max +0.1055。
- 方向范数：||g_theta|| 0.146–0.728；||g_center|| 2.0e-8–4.1e-8（基本为零，单位化后是噪声方向；
  在 t≈1e-6 处 center 实际位移 ~1e-6，可忽略）。
- 事件统计（每层）：raw ≈ 30.0M，dedup ≈ 24.1M，8 槽 t 范围 3.1e-7 … 7.1e-6。
- changed-code（相对父，每事件，folds 合计）：Q ≈ 2.91M–6.82M，K ≈ 0.93M–1.71M；
  **同一层 8 个槽之间几乎相同（个位数差异）** —— 主体差异来自路径原点 theta0 与父部署
  theta_final 之间相差的一个 Adam 步，而非事件阈值本身；所有接受事件同时大量翻 Q/K 码，
  不支持"收益只来自 center"（A-H2 前提不成立）。
- 注意：fold 改善与 holdout/测试窗符号在 5 个接受层中 3 层背离（L1/L8/L15 holdout 转劣），
  测试窗 paired Δ 正负混杂；holdout 只记录未否决（符合规格）。
- calibration API 时间（cache miss）：每层 9.86–10.63s，六层合计 ≈ 61s；
  shard0 在 all6 中 cache hit（0s）。时间风险标注：相对根的新鲜 Attention 校准有额外开销，
  未与根的新鲜校准同机对比；官方 300s 是唯一时间裁决。

## 产物路径

- 构建/配置/验证：`workbench/full_solution/attention-ah1-threshold-events/`
  （`build.py`、`ah1_code.py`、`verify.py`、`config.json`、`verification.json`、`candidate/solution.py`）
- 本地评测：`artifacts/proxy_v3/full_solution/attention-ah1-threshold-events-shard0/`、
  `artifacts/proxy_v3/full_solution/attention-ah1-threshold-events-all6/`
- 逐层 ah1 审计（含 8 个 t、changed-code Q/K 拆分、fold/holdout loss、最终 state）：
  `artifacts/official_eval/cache/proxy-v3-calibration/d712da36e81889c5-attention-*.pt`
  （六个文件，每层一个；最终部署 state 即其中 attention_states）

## 归档与官方状态（v223）

- 归档版本：v223；候选 SHA `d712da36...42990da`；父 SHA `56DC805D...EFCB2BD`。
- 归档前脱离仓库单文件导入检查通过（六 API，孤立目录，SHA 一致）。
- 合法性：逐层 `ah1_*` 审计字段写入 q/k_state，通过 `evaluator/reference_hif4.py` 检查；
  未接受层与父逐位等价（shard0 L0 全零证实 control 干净）。
- 本地总体：72 case mean `+0.003209`、median 0、+28/−32/0=12；5/6 层接受事件，
  机制可达且非等价；5 个接受层中 3 层 holdout 转劣（过拟合迹象，holdout 按计划只记录）。
- 官方状态：`unregistered/NA`，按计划作为单一代表候选交官方裁决。
- 证据：`workbench/full_solution/attention-ah1-threshold-events/`、
  `artifacts/proxy_v3/full_solution/attention-ah1-threshold-events-shard0/` 与 `-all6/`。

## 官方结果（更新）

- **TIMEOUT（>300s）**（2026-09-09 用户回传），无分数，根不变。
- 结论：阈值事件搜索的校准成本（8 槽 × folds + 参考态 ≈ 43 次完整部署路径窗口评估/层）
  在官方侧致命，本地六 shard 的精度信号（mean +0.0032）未能获得官方定价。
  只关闭该实现；同成本类的事件搜索实现重试前必须先降校准成本。
