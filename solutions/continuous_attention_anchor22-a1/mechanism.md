# A22-1：固定 scale 提案 + 完整 R3 对照与回退

登记于 2026-09-07；一个机制、一个配置，见 config.json。官方状态 NA。
隶属 [14199 回传后计划](../../../../docs/superpowers/plans/workpackages/attention-after-14199.md)。

A21-1 官方 14199/244s（相对 R3 −206/+6s）已 REJECTED。归因：A21-1 删除了 R3 旧
训练器，未接受层回退到**训练前栈 B**（无 learned_rotation/learned_center），
36 个回退 case 相对 R3 −0.005979，抵消了 12 个接受 case 的 +0.022872。
本卡分离"移除旧训练"因素：机制为**相同 scale 提案 + 完整 R3 保护**，
不是 A21-1 学习率/门限邻域。

## 固定实现

- 实现父：`solutions/v162_attention_r3-rotation-center_allgates/solution.py`
  SHA `A5C679D7A2B349A879B2019B4A613244F5FEA7050E407B6475B9E29BD1C146DC`。
  保留其 `_a2_train_rotation`、`_a2_true_path_gate_loss`、`_V189_CALIBRATION_ATTENTION`
  前缀（含全部 Linear/V 依赖）不动。
- 校准顺序（每次 attention 校准调用执行）：
  1. 基础栈 B 由 `_V189_CALIBRATION_ATTENTION` 只计算一次。
  2. **P**：用 R3 原始训练（`_a2_train_rotation`，窗口/采样/center/量化/选择规则
     全部不变）和 R3 原始 gate（identity vs rotation 归一化比例）得到完整父状态。
  3. **C**：用 B 坐标执行 A21-1 原训练函数（`_a21_train`，32 步、lr0.01、clip1、
     正则 0.001、cond≤2、epsilon/amax 并列梯度全部不变），提案
     `q_state/k_state.learned_rotation = R0·exp(±S)`。
  4. **A22 gate**：在原最后校准窗口比较 C 与 P 的真实 readout MSE
     （`_a21_gate_loss`：三动态 API + attention 前向）；**严格小于才接受**，
     平局或候选不合法保留 P。
- 未接受层返回完整 P 的 Q/K/V 编码行为，含 learned_rotation 与 learned_center；
  不得只保留基础 multiplier/pair transform，不得按层号/长度/模型建白名单。
- 不吞异常：训练或 gate 的实现错误直接传播记 ERROR，不用吞异常制造全部回退
  的假成功。R3 自身的 gate 语义（比例比较、<1 接受）原样保留。
- 动态 API 无新增候选循环；校准成本为一次旧 R3 训练 + 一次 scale 训练，
  这是本卡明确支付的成本，不称低成本替换。

## 验收

- **强制关闭新候选（gate 第二次调用返回 +inf）时，在合成窗口与真实 L0/L23
  窗口上，输出 state 的全部非审计字段与 R3 官方源码逐位一致**——这是
  "拒绝层对 P 零变化"的直接证明。
- A21-1 提案函数输入/配置不变：同输入生成相同 TQ/TK（独立 GQA 参考复核）。
- 手工指数/scale 梯度对 autograd；B→C 连续 QK 不变；inference 模式可达。
- 记录每层 attempted/accepted/实际回退父身份（parent_arm）、
  P/C 校准 MSE、独立 holdout 误差；本地分解"接受层变化 / 完整父回退层严格零变化"。
- 全部 no-op 不提交，不作为方向饱和证据。

## 评测与官方探索

顺序：最小 smoke/control → shards 0,2 → 完整六片 → OOD 记录 →
fresh default 六 API 计时 → 跨模型记录 → 单文件检查 → 归档。
父已有同 SHA 同协议结果直接复用，身份精确匹配。

本卡新登记**一个**官方机制检验代表：负向 L1<0.02、合法/有限/control/
非 no-op/时间门通过后，本地均值与 split 符号只记录，不据其正负为同路线
排序或拒绝官方检验；负向时标 EXPLORATORY。不沿用 A21-1 已用完的探索名额，
不自动扩到后继卡。

时间用 fresh default 六 API 原模型；预测 <280s 是既有门。额外报告
"预测+24.716s" 单样本残差情景（非新预测）；若该情景 ≥300s，先检查
基础栈/编码/gate 重复计算并减少可消除的重复工作。分项记录旧训练、
新训练、gate 成本。

官方裁决：比 R3 分数高且 <300s 才证明该保护机制有官方增益；低于或等于
R3 则本卡关闭。高于 R3 但低于 A2 时仅为研究候选。相同 SHA 不复测。

## 复现入口

CUDA venv：`build.py` → `verify.py` → `check_math_and_import.py`；
`run.py screen` → `run.py full` → `run.py ood` → `run.py timing` → `run.py cross`。
结果目录 `artifacts/proxy_v3/continuous/attention/anchor22-a1/`；
R3 原始 JSON 逐字节复用，不重跑父 API。
