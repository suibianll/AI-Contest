# A24：父坐标残余互逆变换的直接量化输出误差训练

登记于 2026-09-07；一个机制、一个配置，见 config.json。官方状态 NA。
隶属 [21071 机制证据驱动下一轮计划](../../../../docs/superpowers/plans/workpackages/21071-evidence-driven-research.md)。
用户指令：转向直接优化实际量化 QK／Attention 输出误差，先与旧实现去重，
不再追求 scale 数值更小。父为 **A22-2**（`solutions/continuous_attention_anchor22-a2/solution.py`，
SHA `4686ad81…`，官方 14424/271s）；高分对照 A2（14440/274s）；
低成本对照 R3（14405/238s）。

## 机制

A21/A22/A23 三代残余训练都用 scale 数值代理（可加 amax 比值 / 联合乘积），
A23 证明代理与真实输出脱节（乘积降 45–55% 而 readout 不跟随）。A24 **把残余
训练目标换成真实部署路径输出误差**：

```text
prepared(f)：A2 同款 KV/Q 偶采样子窗口；reference = attention(q_ref, k_ref, v_ref)
             （NVFP4 解码）；v_hat = decode(encode(v_sub))（V 冻结量化）；
             mse_std = MSE(attention(std_q, std_k, std_v), reference)（standard HiF4 归一）
x_q = stack(q)@Rq；x_k = (stack(k)@Rk + c)（P 坐标，含 center）
step: y_q = x_q@ep；y_k = x_k@em
      q_hat = decode(encode(y_q))；k_hat = decode(encode(y_k))   # 真实量化前向
      loss_f = MSE(attention(q_hat, k_hat, v_hat), reference) / mse_std
      d_qhat, d_khat = _m_attention_backward(...)                # 手工 attention 反传
      dL/d ep = x_qᵀ·d_qhat；dL/d em = x_kᵀ·d_khat               # STE：量化视为恒等
      exp_backward + 对称化 + 零迹中心化 + clip1 → Adam → 谱投影（cond≤2）
```

- 目标与 **gate 完全同源**（`_a21_gate_loss` 同款真实 readout MSE；gate 另含
  mse_std 无关的绝对值比较）——消除 A23 的代理失配。
- 32 步 Adam/lr0.01/clip1、零迹 + ±log2/2 谱投影（cond≤2）、
  手工矩阵指数梯度——优化器与 A22-2 相同；**无 scale 正则项**（误差目标
  本身控制目标，复杂度由步数与投影约束）。
- K-center 同步编译 `c_new = c·exp(−S)`；S=0 逐位恢复父；gate 为
  C' vs 完整 A22-2 流程父的真实 readout MSE，严格小于才接受。
- 训练窗口 = `calib_qkv_list[:-1]`（前 4），gate = 最后窗——与父划分相同，
  validation/test 不参与。V 与两个 Linear API 逐位冻结；不吞异常；
  动态 API 无新增候选循环。

## 去重审读（用户指令：先与旧实现去重）

| 对照 | 机制本质 | 与 A24 的区别 | 结论 |
|---|---|---|---|
| v161/v128 Cross-Gram64 per-call 动态精化（官方 timeout，全族关闭） | **动态 API（部署侧）** per-call 小张量精化 | A24 全部计算在**校准侧**，动态 API 只执行编译好的固定矩阵——无 per-call 成本，不在关闭族内 | 不同侧 ✓ |
| v187 Jacobian 坐标敏感度（9167/169s RETAINED） | 一次解析 Jacobian 形成 Q/K importance，静态选择/加权，无迭代训练 | A24 是迭代训练（32 步 Adam）直接最小化真实误差；v187 无残余参数化 | 不同机制 ✓ |
| A2 旧训练（R3 内，14405 组成部分） | **同形目标**（真实输出 MSE/mse_std）+ 手工 attention 反传 | 变量不同（Cayley theta+独立 center vs 对称互逆 exp(±S)+center 编译）、插入点不同（B 坐标 vs P 坐标之后）、部署类不同（正交旋转 vs R3_rot@exp(S) 互逆对）、父不同（R3 14405 vs A22-2 14424 官方线） | 表达空间不同类，非等价 ✓ |
| A21/A22/A23 | scale 数值代理目标 | A24 非代理，直接误差目标；A23 已证代理失配 | 不同目标 ✓ |
| v187 移植族 / +4 scale 窗口 / block-smooth 覆盖率 | 静态 importance/静态 scale 选择 | 无迭代误差训练 | 不同 ✓ |

数学不等价声明：A24 的部署对 `R3_rot@exp(±S)` 不是正交旋转（exp(S) 对称正定），
不能由 A2 的 Cayley 训练在 B 坐标表示出"先接受 R3 再残余修正"的耦合 center
编译路径；插入点在 A22-2 官方父之后，是新的增量参数化。

## 验证清单

- `_m_attention_backward` vs autograd attention 反传（精确）。
- STE 全链：autograd 对"无量化"路径（y=x@ep 直入 forward）的 dL/d ep 对比
  手工链（验证 attention 反传 + einsum 链式；STE 对离散量化本身是登记设计，
  与 A2 同款）。
- S=0 强制回退 = A22-2 官方源码逐位（合成 + 真实 4B 窗口）。
- 互逆转置、GQA 组映射、center 编译恒等、坐标链拦截、V/Linear 冻结、
  inference_mode 可达、fuzz 合同、隔离导入。
- attempted/accepted/changed 计数；QK/logits/probability/QKV 分解仅作解释。

## 评测与官方探索

4B 面板 paired：父 = A22-2 归档源码（4B 基线已由 A23 run 补齐，
72 case mean 0.536715；同 SHA 复用 `--reuse-existing` 语义由 evaluator 身份
检查保证，不重跑父侧）。顺序：shard0 冒烟 → 固定六 shard（72 case）。
判读门：Δmean>0、validation/test 各正、mean(max(−Δgain,0))<0.02。
无本地时间门；api_seconds 记录；官方 300s 唯一硬限。
失败换机制；不在步数/lr/窗口/gate 阈值上扫描。

## 复现入口

CUDA venv：`build.py` → `verify.py` → `check_math_and_import.py` →
`fuzz_check.py`；`run.py screen` → `run.py full`。
结果目录 `artifacts/proxy_v3/continuous/attention/anchor24-a1/`。

## 执行裁决（2026-09-07）：REJECTED_BEFORE_EVALUATION / NO_SUPPORTED_MECHANISM

实现完成后、4B 评测前，验证链暴露一个**数学结构性发现**，按 AGENTS §5
"已确认不可行可提前停止"终止本卡（未跑六 shard、未归档官方候选）：

### 发现：互逆参数化下 STE 误差目标的作用通道精确为零

- 部署 logits = y_q·y_kᵀ = x_q·exp(S)·exp(−S)·x_kᵀ ≡ x_q·x_kᵀ——**未量化
  路径与 S 无关**（exp(S)·exp(−S)=I 的一阶变分精确为零）。
- 残余变换影响输出的**唯一通道是量化码字变化**（块幅度改变 → E6M2/NVFP4
  码字改变 → 解码值改变）。
- STE（量化视为恒等）恰好抹掉这个唯一通道：一阶梯度**精确为零**
  （float64 有限差分仲裁 = 0.0；autograd 对称分量 ~1e-20）。
- float32 实测：`_a24_train` 的 s 仍移动（s_norm 8.69）——这是 **Adam 把
  float32 数值噪声（~1e-8）放大为 lr 级随机游走**的伪梯度；训练窗口 loss
  下降 6.6% 属窗口内噪声过拟合，独立 gate 窗口正确拒绝
  （candidate 4.560e-4 > parent 4.328e-4）。

### 工具链副产物（重要，供后续卡使用）

- **float32 的 `torch.matrix_exp` autograd 数值上是坏的**（与 float64
  有限差分仲裁差 O(1)）；float64 下 autograd 与 Daleckii–Krein 解析式
  一致到 5e-15（含投影产生的重复特征值场景）。任何用 float32 matrix_exp
  autograd 做参考的梯度验证都不可信；须用 float64 参考或有限差分。
- `_a21_exp_backward` 经 float64 仲裁**数学完全正确**（重复特征值含）。
- A22-2/A23 的验证不受影响（它们不用 autograd 训练；此前的
  gradient-vs-autograd 检查对象是 attention/amax 链，无 matrix_exp）。

### 对路线的含义

"直接优化实际量化输出误差"在互逆参数化下的合法实现只剩：
(a) 穿过量化的真实梯度（离散码级方法——GPTQ 式坐标优化或码字差代理，
    需要与旧关闭族去重后新登记）；
(b) gate 本身已是真实误差选择器（A22-2 = scale 代理提议 + 真实误差 gate），
    改进空间在**提议质量**而非 gate；
(c) 放弃互逆参数化回到独立变换（A2 已存在，非新机制）。
本卡关闭不否定"直接优化输出误差"的方向，只否定 **STE 连续梯度**这一实现。
