# A22-2 执行记录：父坐标残余互逆变换，本地新最高 0.783515

2026-09-07，Attention 侧持续优化（总计划 §2/§3，工作包 attention-after-14199 §4）。
run_id `anchor22-a2`；候选 SHA256
`4686ad817128d60e1a1648e0e14793692097ee8171fac2bc5202466d5787c0b7`；
实现父 R3 `a5c679d7a2b349a879b2019b4a613244f5fea7050e407b6475b9e29bd1c146dc`
（14405/238s）。评测器 evaluator/eval.py（eval-v3/proxy-v3），proxy-v2 dense cache，
CUDA，attention-only，6 shard；R3 baseline 按路径+SHA 复用，未重跑父 API。

## 背景

A22-1（完整 R3 回退保护）本地通过全部门（Δmean +0.005786、负向 L1 0.0000329）
但仅 3/24 层接受，21/24 提案被完整 P 拒绝——A22-2 的启动条件。
A22-1 官方未回传，本卡按工作包 §4 仅 LOCAL_RESEARCH，不登记官方探索。

## 实现

残余互逆变换：保留 P 的 learned_rotation/learned_center，训练残余
S（`Qnew=ZQ·exp(S)`、`Knew=ZK·exp(−S)`，S=0 逐位恢复 P）；部署编译
`Rq_new=Rq·exp(S)`、`Rk_new=Rk·exp(−S)`、`c_new=c·exp(−S)`（center 同步）。
分母为 S=0 完整父坐标 amax；训练参数与 A21-1 全同；gate 对完整 P 严格小于。

实现事故修复（均先于评测）：torch 2D@3D 广播把 `c@em` 解释为 batch 矩阵乘
（改为 `c.unsqueeze(-2)@em`）；identity 臂 `torch.eye(dim)` 缺 kh 组维（expand）。
两次修复后 build/verify 全过，随后未再改动源码。

## 验证

- S=0 平局回退与官方 R3 逐位一致（合成 34 字段、L0 34、L23 43）。
- 编译恒等式 `(UK@Rk+c)@em == UK@(Rk@em)+c@em`；S=0 编译矩阵与父逐位相等。
- 坐标链拦截、V 冻结、R3 前缀、梯度（max 2.62e-6）、隔离导入、inference 可达。

## 结果（本地误差指标，非官方分）

| 指标 | ID 48 case | OOD 48 case |
|---|---:|---:|
| Δmean | +0.021296052 | +0.017128766 |
| 负向 L1 | 0.000103194 | 0.000673254 |
| 正/负/相同 | 23/1/24 | 21/3/24 |
| validation Δmean | +0.018684347 | +0.016834301 |
| test Δmean | +0.023907758 | +0.017423230 |

12/24 层接受（L3/6/7/11/12/13/14/15/18/20/21/23）；最大层增益 L12 +0.106、
L20 +0.081、L23 +0.066。唯一负 case L7 −0.004953。Δgap +0.004167（<0.01）；
最差 OOD case L23 −0.0270 记录。

## 本地最高与时间

- default 120 case attention_mean **0.783515197** > 旧本地最高 0.773281（A1）
  → 按用户当前指令触发归档 + commit/push。
- fresh default：api_total 105.526s（A_calib 99.859s）；官方时间预测
  **235.323885s < 280s**；+24.716s 情景 260.039885s < 300s。
- GPT-2 compact 0.477294（仅记录）。

## 裁决

**LOCAL_RESEARCH_COMPLETED / NEW_LOCAL_HIGHEST / official NA。**
候选归档 `solutions/continuous_attention_anchor22-a2/`。
A22-1 官方回传后由用户/协调者决定本卡官方探索资格。
