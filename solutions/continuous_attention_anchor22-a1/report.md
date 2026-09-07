# A22-1 执行结果：本地全门通过，登记官方探索代表

2026-09-07。隶属 [14199 回传后计划](../../../docs/superpowers/plans/workpackages/attention-after-14199.md) §3。
候选源码 SHA256：`25310c6e36c41ca0dd7a716c29946ede9c7e030af7c4329dc2f29d732bd92447`。
实现父 R3 14405/238s（SHA `A5C679D7…`）；高分对照 A2 14440/274s 不变。
A21-1 官方 14199/244s REJECTED 不替换任何父；根 solution.py 仍为 v189。

## 机制与实现

与 A21-1 唯一的区别是对照与回退对象：A21-1 的 gate 比较的是"训练前栈 B"，
未接受层丢失 R3 旧训练（learned_rotation/learned_center），官方 −206。
A22-1 保留完整 R3 源码前缀（`_a2_train_rotation`、`_a2_true_path_gate_loss`、
`_a2_normal` 逐字节不动），校准时先按 R3 原始训练和选择逻辑得到完整父 P
（parent_arm：15 层 rotation、9 层 identity），再用 A21-1 原训练函数
（B 坐标、32 步、lr0.01、clip1、正则0.001、cond≤2，全部不变）得到提案 C，
最后在原最后校准窗口比较 C 与 P 的真实 readout MSE，严格小于才接受。
不吞异常；动态 API 无新增候选循环；V 与两个 Linear API 冻结。

## 关键验收：回退层对完整 R3 严格零变化

强制关闭新候选（gate 第二次调用返回 +inf）时，输出 state 的全部非审计字段
与官方 R3 源码同输入输出**逐位一致**：

- 合成窗口（96×896/128，3 窗口）：40 字段 bit-identical
- 真实缓存 L0：34 字段；L23：43 字段

同时验证部署坐标链（拦截 `_dense_to_hif4` 输入 = `_a1_stack_transform`
+ learned_rotation（+ K learned_center））、V 五字段与基础栈逐位一致、
inference_mode 可达、隔离单文件六 API 导入、手工梯度 vs autograd
（max 3.219e-6）。互逆误差 max 2.563e-5（与 A21-1 相同量级）。

## 同协议结果（本地误差指标，不是官方分数）

| 指标 | ID 六片 (48 case) | OOD 六片 (48 case) |
|---|---:|---:|
| R3 mean gain | 0.775557865 | 0.769095528 |
| 候选 mean gain | 0.781343521 | 0.771936650 |
| Δmean | **+0.005785656** | **+0.002841122** |
| 负向 L1 | **0.000032880** | 0.000284406 |
| 总 L1（记录） | 0.005851416 | — |
| 正/负/相同 case | 5/1/42 | 4/2/42 |
| validation Δmean | +0.003830893 (2/1/21) | +0.003669317 |
| test Δmean | +0.007740418 (3/0/21) | +0.002012927 |

- 配对 gap：候选 +0.009406870，R3 +0.006462336，**Δgap +0.002944534**
  （低于 0.01 提示阈值；OOD 本身只作风险诊断，不作否决门）。
- 变化层与校准 gate 精确一致：仅 L8/L12/L23 部署提案，其余 21 层回退完整 R3
  （ID 上严格零变化）。接受层 ID：L8 +0.000901、L12 **+0.087148**、L23 +0.050806；
  OOD 上同层也全为正：+0.001169 / +0.056159 / +0.010859。
- 24/24 层 attempted；3 accepted。gate 比值：L8 0.9955、L12 0.8355、L23 0.9877。
- 逐 case、分组与全部 SHA 见 manifest.json 与 calibration-audit.json。

对照 A21-1（同 48 case）：Δmean +0.001234→**+0.005786**（4.7 倍），
负向 L1 0.007585→**0.0000329**（230 倍改善），正/负/相同 16/18/14→5/1/42。
"移除旧训练损失"因素被分离：回退层从 −0.005979 变为严格 0，
接受层收益（+0.0229 组均值）由 L12/L23 的更大增量承担。

## 成本与时间门

fresh default（168 W + 168 A + 24 attention + 120×3 动态）实测：

| API | 秒 |
|---|---:|
| W calibration | 0.689809 |
| dynamic A | 1.461496 |
| Attention calibration | 113.530390 |
| dynamic Q | 3.043307 |
| dynamic K | 1.995639 |
| dynamic V | 1.538619 |

API total **122.259262s**，wall 169.529423s。
六 API 模型官方预测 **239.849602s < 280s** ✓；
`预测+24.716s` 单样本残差情景 **264.565602s < 300s** ✓（情景是敏感性，非保证）。
A_calib 相对 A21-1（75.714s）增加 37.8s，即一次完整 R3 训练 + 两次 gate 的
预注册成本。fresh default attention_mean **0.768434864**（120 case），
高于 A21-1 的 0.766631，仍低于本地最高参考 A1 0.773281（同口径 default
120 case），该口径不与 48 case eval-v3 混排。
GPT-2 compact 4 case：0.489213（仅记录，不提供晋级/否决依据）。

## 裁决

**READY_FOR_OFFICIAL_EXPLORATION / official unregistered/NA。**
本卡登记的唯一官方机制检验代表：Δmean>0、负向 L1 0.000033<0.02、
validation/test 双 split 正、合法性/有限/control/非 no-op/时间门全部通过；
本地符号只记录。官方裁决规则：比 R3 分数高且 <300s 才证明保护机制有官方
增益；≤R3 则本卡关闭；高于 R3 低于 A2 仅为研究候选。相同 SHA 不复测。

候选独立归档至 `solutions/continuous_attention_anchor22-a1/solution.py`。
官方提交由协调者/用户执行；本侧不重复提交相同 SHA。

## 下一步

等待官方回传期间，可本地准备 **A22-2（父坐标增量互逆变换）**：其启动条件
"多数提案仍被完整 P 拒绝"已满足（21/24 回退，其中 11 层 parent_mse 与
candidate_mse 差距 <20%）。A22-2 保留 P 的 rotation/center，仅学习残余
S=0 恢复父，K-center 同步 `c·exp(−S)`；须先独立登记，不把待回传源码
升级为官方父。不在失败配置周围扫参数。

## 复现入口

CUDA venv：`build.py` → `verify.py` → `check_math_and_import.py`；
`run.py screen` → `run.py full` → `run.py ood` → `run.py timing` → `run.py cross`；
`audit_calibration.py`（24 层 gate 审计）。R3 原始 JSON 逐字节复用（SHA 校验），
未重跑父 API。结果目录：`artifacts/proxy_v3/continuous/attention/anchor22-a1/`。
