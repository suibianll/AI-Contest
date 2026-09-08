# A22-2 执行结果：本地新最高（default 0.783515 > 0.773281），按指令归档提交

2026-09-07。隶属 [14199 回传后计划](../../../docs/superpowers/archive/plans/attention-after-14199.md) §4。
候选源码 SHA256：`4686ad817128d60e1a1648e0e14793692097ee8171fac2bc5202466d5787c0b7`。
实现父 R3 14405/238s（SHA `A5C679D7…`）。A22-1（anchor22-a1，官方 NA）已先行完成。
本卡为 **LOCAL_RESEARCH**：A22-1 未回传，官方探索未登记、不提交官方；
按用户当前指令"超过本地最高精度时提交并推送代码"，本地最高被超过后执行
归档 + commit/push（origin-ssh），官方提交决定权仍在用户/协调者。

## 机制

在完整 R3 父坐标上的残余互逆变换：保留 P 的 learned_rotation/learned_center，
仅学习残余 S（无 R0 初始化）。训练 `Qnew = ZQ·exp(S)`、`Knew = ZK·exp(−S)`
（ZQ=UQ·Rq，ZK=UK·Rk+c）；部署编译 `Rq_new = Rq·exp(S)`、`Rk_new = Rk·exp(−S)`、
**`c_new = c·exp(−S)`**（K-center 与 rotation 同步变换，遗漏会破坏父连续 QK）。
scale 目标/预算与 A21-1 相同，分母改为 S=0 完整父坐标 amax。
gate 仍对完整 P（真实 readout MSE，严格小于才接受）；训练函数与 A22-1 同构。

## 关键验收

- **S=0 平局逐位恢复 P**：强制残余为零时 gate 平局回退，输出与官方 R3 源码
  逐位一致（合成 34 字段、L0 34、L23 43，全部非审计字段）。
- **部署编译恒等式**：`(UK@Rk + c)@em == UK@(Rk@em) + c@em` 数值验证通过；
  S=0 时编译矩阵与父矩阵逐位相等（rq@I=rq 等）。
- 部署坐标链拦截（stack → compiled rotation → compiled center）逐位一致；
  V 冻结逐位 control；R3 前缀逐字节；梯度 vs autograd max 2.62e-6；
  隔离单文件六 API 导入；inference 模式可达。
- 修复两个实现 bug 后通过全部检查：2D@3D 广播把 c@em 变成 batch 矩阵乘
  （改为 c.unsqueeze(-2)@em），identity 臂 eye 缺 kh 组维（expand 修复）。

## 同协议结果（本地误差指标，不是官方分数）

| 指标 | ID 六片 (48 case) | OOD 六片 (48 case) |
|---|---:|---:|
| R3 mean gain | 0.775557865 | 0.769095528 |
| 候选 mean gain | **0.796853917** | 0.786224294 |
| Δmean | **+0.021296052** | **+0.017128766** |
| 负向 L1 | **0.000103194** | 0.000673254 |
| 正/负/相同 case | 23/1/24 | 21/3/24 |
| validation Δmean | +0.018684347 (11/1/12) | +0.016834301 |
| test Δmean | +0.023907758 (12/0/12) | +0.017423230 |

- 配对 gap：候选 +0.010629623，R3 +0.006462336，Δgap +0.004167286
  （低于 0.01 提示阈值；OOD 只诊断不否决）。最差 OOD case L23 −0.0270 仅记录。
- **12/24 层接受**（L3/6/7/11/12/13/14/15/18/20/21/23），与校准 gate 精确一致；
  其余 12 层相对 R3 严格零变化。最大层增益：L12 +0.106191、L20 +0.081058、
  L23 +0.065570、L21 +0.052390、L15 +0.036818。唯一负 case：L7 −0.004953。
- 对照 A22-1：Δmean +0.005786→**+0.021296**（3.7 倍），接受层 3→12，
  ID mean 0.781344→0.796854。残余参数化保留父变换后再优化，
  在 R3 已有 rotation 的层上优于替换式参数化。

## 本地最高（default 120 case 口径，不与 48 case 混排）

| 候选 | default attention_mean |
|---|---:|
| **A22-2（本轮）** | **0.783515197** |
| A1 deployed-aligned（旧最高） | 0.773281 |
| A22-1 | 0.768434864 |
| A21-1 | 0.766631390 |

**新本地最高 +0.010234**，触发用户指令的提交推送条件。

## 成本与时间门

fresh default 实测：W 0.586700 / act 1.590874 / **attention calib 99.859072** /
Q 1.529490 / K 1.131314 / V 0.828742；api_total **105.526193s**（比 A22-1
122.259s 更快），wall 135.148472s。六 API 模型官方预测 **235.323885s < 280s** ✓；
`+24.716s` 残差情景 **260.039885s < 300s** ✓。
GPT-2 compact 4 case：0.477294（仅记录）。

## 裁决与下一步

**LOCAL_RESEARCH_COMPLETED / NEW_LOCAL_HIGHEST / official NA。**
候选独立归档 `solutions/continuous_attention_anchor22-a2/solution.py`。
等待 A22-1 官方回传；回传后由用户/协调者决定本卡官方探索资格（不自动继承
A22-1 名额）。若 A22-1 SCORE_BEST 晋级，本卡需基于新父重新验证。
0.9 目标继续：48 case ID mean 0.7969（+0.0213/轮），default 0.7835；
下一机制候选：A22-3（输出失配目标，条件卡）或新一轮残余自由度
（如分组 S / 每组独立预算——须先登记新卡，不在本配置上扫参数）。

## 复现入口

CUDA venv：`build.py` → `verify.py` → `check_math_and_import.py`；
`run.py screen` → `run.py full` → `run.py ood` → `run.py timing` → `run.py cross`；
`audit_calibration.py`。R3 原始 JSON 逐字节复用（SHA 校验），未重跑父 API。
结果目录：`artifacts/proxy_v3/continuous/attention/anchor22-a2/`。
