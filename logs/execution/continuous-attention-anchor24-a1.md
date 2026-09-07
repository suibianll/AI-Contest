# A24 执行记录：STE 误差目标在互逆参数化下无作用通道，提前停止

2026-09-07，Attention 侧（21071 下一轮计划，用户指令"转向直接优化实际量化
QK／Attention 输出误差，先与旧实现去重"）。run_id `anchor24-a1`；
候选源码 SHA256（构建后）
`a234885477df44e836128e93d08dbe4d206b7cfbe95d16e740f04abf669fb77c`（修复
device bug 后 rebuild，SHA 以 verification.json 为准）；父 A22-2
`4686ad81…`（14424/271s）。

## 机制（已实现）

A22-2 全套结构不变（B 栈 → R3 原始训练 → 完整父 P → P 坐标残余 exp(±S) →
真实 readout gate 严格小于），只把残余训练目标换成**真实部署路径输出 MSE**
（A2 同款 mse_std 归一、KV/Q 偶采样、STE 手工梯度、无 scale 正则）。
去重审读完成（v161 动态精化=部署侧、v187=静态 importance、A2=同目标不同
参数化/插入点/父、A21/A22/A23=代理目标——均不等价）。

## 结构性发现（本卡核心产出）

验证链发现并经 float64 有限差分仲裁确认：

1. **互逆参数化 exp(±S) 在未量化路径上与恒等无异**：logits =
   x_q·exp(S)·exp(−S)·x_kᵀ ≡ x_q·x_kᵀ，一阶变分精确为零。
2. 残余变换影响输出的唯一通道 = **量化码字变化**；STE（量化恒等近似）
   恰好抹掉该通道 → **误差目标的 STE 梯度信号精确为零**。
3. float32 实测：s_norm=8.69 的移动是 **Adam 把 float32 数值噪声放大为
   lr 级随机游走**的伪梯度；训练窗 loss 降 6.6% 为窗口内噪声过拟合，
   独立 gate 窗口正确拒绝（4.560e-4 > 4.328e-4，layer 0 实测）。
4. 工具链：**float32 的 `torch.matrix_exp` autograd 数值上是坏的**
   （float64 有限差分仲裁差 O(1)）；`_a21_exp_backward` 经 float64 仲裁
   数学完全正确（重复特征值含，一致到 5e-15）。

## 裁决

**REJECTED_BEFORE_EVALUATION / NO_SUPPORTED_MECHANISM**（AGENTS §5
"已确认不可行可提前停止"；未跑六 shard、无官方候选）。依据：机制的一阶
作用通道被参数化结构精确清零，评测只能产出 no-op 或噪声结果。

关闭边界：只否定 **互逆参数化 + STE 连续梯度** 的组合；不否定
(a) 穿过量化的离散/码级真实梯度方法（须与旧关闭族去重后新登记）、
(b) A22-2 已验证的"scale 代理提议 + 真实误差 gate"结构、
(c) 独立（非互逆）变换参数化（A2 已存在）。

产物：workbench/continuous_attention/anchor24-a1/
（mechanism 含发现附录、trainer/build/verify/run/fuzz_check、
verification.json、math-and-import.json）。
