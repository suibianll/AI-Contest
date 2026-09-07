# A22-1 执行记录：固定 scale 提案 + 完整 R3 回退保护

2026-09-07，Attention 侧持续优化（总计划 §2/§3，工作包 attention-after-14199 §3）。
run_id `anchor22-a1`；候选 SHA256
`25310c6e36c41ca0dd7a716c29946ede9c7e030af7c4329dc2f29d732bd92447`；
实现父 R3 `a5c679d7a2b349a879b2019b4a613244f5fea7050e407b6475b9e29bd1c146dc`
（14405/238s）。评测器 evaluator/eval.py（eval-v3/proxy-v3），proxy-v2 dense cache，
CUDA，attention-only，6 shard；R3 baseline 按路径+SHA 复用原 JSON，未重跑父 API。

## 背景

A21-1 官方 14199/244s REJECTED（相对 R3 −206/+6s）。本卡执行工作包 §3
A22-1：保持 A21-1 固定 scale 提案不变，gate 由"C vs B（训练前栈）"改为
"C vs 完整 R3 父 P"，未接受层回退完整 P（含 learned_rotation/learned_center）。

## 执行过程

1. `build.py`：R3 源码保留至最终包装之前（含旧训练器/gate/_a2_normal），
   拼 A1 `_a1_state_on_device`/`_a1_stack_transform` 与新 trainer；AST 检查通过。
2. `check_math_and_import.py`：端到端目标梯度 max 1.014e-6；隔离单文件六 API 导入 PASS。
3. `verify.py`（GPU 锁）：独立 GQA 四几何/互逆、手工 exp+scale 梯度 vs autograd
   （max 3.219e-6）、R3 前缀逐字节、**强制拒绝候选 = 官方 R3 逐位一致**
   （合成 40 字段、L0 34、L23 43）、部署坐标链拦截、V 逐位 control、
   inference_mode 可达。全部 PASS。
4. 一次 screen 运行遭遇 CUDA OOM（另一代理 GPU 任务同期运行）；锁空闲后
   重试通过。此后 screen → full → ood → timing → cross 均在共享 gpu.lock 下串行完成。
5. `audit_calibration.py`：24 层校准审计，attempted 24/24，accepted [8,12,23]，
   parent_arm rotation 15 / identity 9（与评测变化层精确一致）。

## 结果（本地误差指标，非官方分）

| 指标 | ID 48 case | OOD 48 case |
|---|---:|---:|
| Δmean | +0.005785656 | +0.002841122 |
| 负向 L1 | 0.000032880 | 0.000284406 |
| 正/负/相同 | 5/1/42 | 4/2/42 |
| validation Δmean | +0.003830893 | +0.003669317 |
| test Δmean | +0.007740418 | +0.002012927 |

- 配对 Δgap +0.002944534（<0.01 提示阈值；OOD 只诊断不否决）。
- 变化层仅 L8/L12/L23；接受层 ID +0.000901/+0.087148/+0.050806，
  OOD 同层 +0.001169/+0.056159/+0.010859；其余 21 层相对 R3 严格零变化。
- 对照 A21-1：Δmean +0.001234→+0.005786，负向 L1 0.007585→0.0000329；
  "移除旧训练损失"被完整父回退消除。

## 时间与成本

fresh default 六 API：W 0.689809 / act 1.461496 / **attention calib 113.530390** /
Q 3.043307 / K 1.995639 / V 1.538619；api_total 122.259262s。
官方时间预测 **239.849602s < 280s**；`+24.716s` 残差情景 264.565602s < 300s。
A_calib 相对 A21-1 +37.8s 为预注册的旧 R3 训练 + gate 成本。
fresh default attention_mean 0.768434864（120 case；A1 参考最高 0.773281，
A21-1 0.766631；口径不与 48 case 混排）。GPT-2 compact 0.489213（仅记录）。

## 裁决与状态

**READY_FOR_OFFICIAL_EXPLORATION / official unregistered/NA。**
本卡登记的唯一官方机制检验代表通过全部前置门（负向 L1<0.02、双 split 正、
合法/有限/control/非 no-op/时间预测门）；本地符号仅记录。
候选归档 `solutions/continuous_attention_anchor22-a1/`（archive SHA 同上）。
官方提交由协调者/用户执行；不重复提交相同 SHA。

A22-2 启动条件已满足（21/24 提案被完整 P 拒绝）；等待官方回传期间仅本地
准备，不把待回传源码升级为官方父，不在失败配置周围扫参数。

产物：workbench/continuous_attention/anchor22-a1/（mechanism/config/manifest/
report/calibration-audit/verification/math-and-import）；
artifacts/proxy_v3/continuous/attention/anchor22-a1/{id,ood,timing,cross}。
