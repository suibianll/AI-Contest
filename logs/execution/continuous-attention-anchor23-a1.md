# A23 执行记录：Q/K 乘积目标 4B paired 负向，目标卡关闭

2026-09-07，Attention 侧（21071 下一轮计划 §5，4B 面板）。
run_id `anchor23-a1`；候选 SHA256
`8714ac2a044779465c5e406ef0768be7071ac626f3a2171cdc5083350be92dbf`；
父 A22-2 `4686ad81…`（官方 14424/271s）。评测器 evaluator/eval.py（eval-v3），
cache `qwen3.5-4b-proxy-v2.pt`，CUDA，attention-only，6 shard 72 case，
`--baseline-solution` 指向 A22-2 归档源码（**父 4B 基线由本次 paired run
首跑补齐**，此前无 anchor 系 4B 运行）。

## 机制

只换残余训练目标：Q/K 联合块 scale 乘积
`mean_{f,g,b}[a_Q(s)·a_K(s)/max(a_Q(0)·a_K(0),1e-12)] + 1e-3·mean(S²)`，
其中 a_Q(g,b) 跨（组 g 全部 Q heads × tokens）对 head 内第 b 个 64 块
的 amax² 均值，a_K(g,b) 跨 tokens；分母为 S=0 完整父坐标聚合。
保留 A22-2 全部其他结构（B 栈 → R3 原始训练 → 完整父回退 → P 坐标残余
→ 真实 readout gate 严格小于）。dim%64≠0 的布局（仅 fuzz）退化为组级聚合。

## 验证

乘积梯度 vs autograd（3 几何 max 1.5e-8；回退 3.7e-9）；Q×c/K÷c 乘积
不变性 PASS；S=0 强制回退 = A22-2 官方源码逐位（合成 37 字段、4B L0 45、
L22 43）；frozen prefix/V/Linear 冻结；inference_mode；隔离导入。
fuzz：1 失败与 A22-2 父逐字相同（mixed-q 校准；官方合同不含该输入；
R3 靠吞异常存活）——继承行为非回归。

## 4B paired 结果（72 case，A23 − A22-2）

| 指标 | 全部 | validation | test |
|---|---:|---:|---:|
| Δmean | **−0.004750** | +0.000753 | **−0.010253** |
| 负向 L1 | 0.005676 | — | — |
| 正/负/相同 | 10/14/48 | 6/6/24 | 4/8/24 |

A22-2 4B 基线 mean 0.536715；A23 0.531965。层级：L0/L1/L5/L15 两侧 gate
均拒绝 → 逐位零变化（12×4 case）；**L8 gate 接受但 9/12 负向（−0.0290，
最后单校准窗口 gate 过拟合）**；L22 +0.0005（7/5）。

## 裁决

**LOCAL_NEGATIVE / OBJECTIVE_CLOSED / official NA。**
专项门 FAIL（Δmean<0、test split 负）。行为与预登记失败模式一致：
乘积比降 45–55% 而真实 readout 不跟随。按工作包 §5 分支关闭此目标；
不扫 loss 权重/窗口/gate 阈值；不扩写为附加式残余族关闭（A22-2 可加
目标有官方 +19）。A22-2 父线不变，本侧无待官方包。
下一卡（实际量化 QK 目标）须先与旧 Jacobian/动态 Gram 关闭族去重。

产物：workbench/continuous_attention/anchor23-a1/（全套）；
artifacts/proxy_v3/continuous/attention/anchor23-a1/id/。
