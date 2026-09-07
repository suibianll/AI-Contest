# A22-2：父坐标上的增量互逆变换（保留 P，仅学习残余 S）

登记于 2026-09-07；一个机制、一个配置，见 config.json。官方状态 NA。
隶属 [14199 回传后计划](../../../../docs/superpowers/plans/workpackages/attention-after-14199.md) §4。
**启动条件已满足**：A22-1 中 21/24 提案被完整 P 拒绝（多数提案被拒绝）。

## 与 A22-1 的关系

A22-1 的提案在 B 坐标上以固定 Hadamard R0 起步（learned_rotation = R0·exp(±S)），
替换 P 的旧变换；多数层无法击败完整 P。A22-2 是**新的残余参数化与初始化**：
保留 P 的 rotation/center，只学习残余 exp(±S)。不是把 A22-1 的失败配置
重跑一遍，也不扫参数。

## 固定实现

- 设 P 最终编码前坐标 `ZQ = UQ·Rq`、`ZK = UK·Rk + c`；P 的 rotation 臂取
  `Rq = Rk = parent.learned_rotation`（R3 部署共享矩阵）、`c = parent.learned_center`；
  identity 臂 `Rq = Rk = I`、`c = 0`。UQ/UK 为 `_a1_stack_transform` 的 B 栈输出。
- 训练 `Qnew = ZQ·exp(S)`、`Knew = ZK·exp(−S)`；**S=0 必须逐位恢复 P**。
- 部署编译 `Rq_new = Rq·exp(S)`、`Rk_new = Rk·exp(−S)`、**`c_new = c·exp(−S)`**；
  数学上 `(UK·Rk + c)·exp(−S) = UK·Rk_new + c_new`，center 同步变换不可遗漏。
  identity 臂 `c_new = 0`，不部署 learned_center（部署 0 或不部署均恒等）。
- scale 目标/预算与 A21-1 相同：每 role 按实际连续 64 块 amax 比例平方、
  等权 fold、Q/K 块均值相加、epsilon=1e-12、amax 并列极值均分次梯度、
  32 步 Adam/lr0.01/clip1/正则0.001、cond≤2、零迹与 ±log2/2 谱投影；
  **分母改为该卡 S=0 的完整父坐标 amax**（不是 B 坐标 amax）。
  不引入敏感度加权或新量化器。
- gate 仍对完整 P：原最后校准窗口比较 C'（P 坐标 + exp(±S)）与 P 的真实
  readout MSE，严格小于才接受，平局或候选不合法保留 P。
- 不吞异常；动态 API 无新增候选循环；校准成本为一次旧 R3 训练 + 一次残余
  scale 训练 + 4 次 gate 评估（与 A22-1 同量级）。
- V 与两个 Linear API 冻结；单文件六 API 契约不变。

## 本地验证（本卡先行为）

- **S=0 平局回退**：强制训练返回 S=0（tq=Rq、tk=Rk、c_new=c）→ gate 平局
  → 输出 = P = 官方 R3 逐位一致（复用 A22-1 的逐位对比协议）。
- **接受路径坐标链**：拦截 `_dense_to_hif4` 输入 = `_a1_stack_transform`
  + `learned_rotation(tq/tk)` +（K）`learned_center(c_new)`。
- K-center 同步检查：c_new 的显式数值对照（防遗漏 center 变换）。
- 手工梯度 vs autograd；互逆误差；inference 模式可达；隔离导入。

## 官方探索规则（本卡独立登记）

**A22-1 未回传期间本卡仅 LOCAL_RESEARCH，不提交官方。** A22-1 回传后：
若 A22-1 SCORE_BEST 成为新父，本卡须基于新父重新验证再申请；
若 A22-1 REJECTED，本卡是否获得官方探索代表资格由用户/协调者单独批准，
不自动继承 A22-1 名额。禁止在等待期间把本地候选冒充官方父。

## 复现入口

CUDA venv：`build.py` → `verify.py` → `check_math_and_import.py`；
`run.py screen` → `run.py full` → `run.py ood` → `run.py timing` → `run.py cross`。
结果目录 `artifacts/proxy_v3/continuous/attention/anchor22-a2/`；
R3 原始 JSON 逐字节复用。
